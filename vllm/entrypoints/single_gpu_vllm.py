import os
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.distributed

from vllm import LLM, LLMEngine
from vllm.attention import AttentionMetadata
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VisionLanguageConfig)
from vllm.engine.arg_utils import EngineArgs
from vllm.executor.gpu_executor import GPUExecutor
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.model_executor import SamplingMetadata, set_random_seed
from vllm.model_executor.model_loader import get_model
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter, CudaMemoryProfiler
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import (BatchType, ModelRunner,
                                      _prepare_fake_inputs, LORA_WARMUP_RANK)
from vllm.worker.worker import Worker, raise_if_cache_size_invalid
from vllm.sampling_params import SamplingParams
import gc


class SingleGPUModelRunner(ModelRunner):

    def load_model(self) -> None:
        """costa: needed to specify the device of `CudaMemoryProfiler`"""
        with CudaMemoryProfiler(device=self.device_config.device) as m:
            self.model = get_model(
                model_config=self.model_config,
                device_config=self.device_config,
                load_config=self.load_config,
                lora_config=self.lora_config,
                vision_language_config=self.vision_language_config,
                parallel_config=self.parallel_config,
                scheduler_config=self.scheduler_config,
            )

        self.model_memory_usage = m.consumed_memory
        print(f"Loading model weights took "
              f"{self.model_memory_usage / float(2**30):.4f} GB")
        print("haha")

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, SamplingMetadata,
               Set[LoRARequest], LoRAMapping, torch.Tensor]:
        prefill_reqs = []
        decode_reqs = []
        for seq_group_meta in seq_group_metadata_list:
            if seq_group_meta.is_prompt:
                prefill_reqs.append(seq_group_meta)
            else:
                decode_reqs.append(seq_group_meta)

        # Prepare input tensors.
        (
            input_tokens,
            input_positions,
            prefill_attn_metadata,
            prompt_lens,
            subquery_lens,
            lora_index_mapping,
            lora_prompt_mapping,
            lora_requests,
            multi_modal_input,
            slot_mapping,
        ) = self._prepare_prompt(prefill_reqs)
        (
            decode_input_tokens,
            decode_input_positions,
            decode_attn_metadata,
            decode_lora_index_mapping,
            decode_lora_prompt_mapping,
            decode_lora_requests,
            decode_slot_mapping,
        ) = self._prepare_decode(decode_reqs)
        sampling_metadata = SamplingMetadata.prepare(seq_group_metadata_list,
                                                     prompt_lens,
                                                     subquery_lens,
                                                     self.device,
                                                     self.pin_memory)

        if not self.scheduler_config.chunked_prefill_enabled:
            assert (len(prefill_reqs) and len(decode_reqs)) == 0

        num_prefills = len(prompt_lens)
        num_prefill_tokens = len(input_tokens)
        num_decode_tokens = len(decode_input_tokens)

        # Coalesce tensors. Note that attn_metadata is currently not
        # coalesced for simplicity.
        input_tokens.extend(decode_input_tokens)
        input_positions.extend(decode_input_positions)
        slot_mapping.extend(decode_slot_mapping)
        lora_index_mapping.extend(decode_lora_index_mapping)
        lora_prompt_mapping.extend(decode_lora_prompt_mapping)
        lora_requests.update(decode_lora_requests)

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)

        if self.lora_config:
            lora_mapping = LoRAMapping(
                lora_index_mapping,
                lora_prompt_mapping,
            )
        else:
            lora_mapping = None

        # Broadcast the metadata.
        # If batch contains both prefill and decode, it sends 2 broadcasts.
        # If it only contains 1 type, it triggers a single broadcast.
        if (prefill_attn_metadata is not None
                and decode_attn_metadata is not None):
            batch_type = BatchType.MIXED
        elif prefill_attn_metadata is not None:
            batch_type = BatchType.PREFILL
        else:
            batch_type = BatchType.DECODE

        metadata_dict = {
            "input_tokens": input_tokens,
            "input_positions": input_positions,
            "selected_token_indices": sampling_metadata.selected_token_indices,
            "lora_requests": lora_requests,
            "lora_mapping": lora_mapping,
            "multi_modal_input": multi_modal_input,
            "num_prefill_tokens": num_prefill_tokens,
            "num_decode_tokens": num_decode_tokens,
            "slot_mapping": slot_mapping,
            "num_prefills": num_prefills,
            "batch_type": batch_type,
        }
        if prefill_attn_metadata is not None:
            metadata_dict.update(prefill_attn_metadata.asdict_zerocopy())
        else:
            assert decode_attn_metadata is not None
            metadata_dict.update(decode_attn_metadata.asdict_zerocopy())

        attn_metadata = AttentionMetadata(
            num_prefills=num_prefills,
            slot_mapping=slot_mapping,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            prefill_metadata=prefill_attn_metadata,
            decode_metadata=decode_attn_metadata,
            kv_cache_dtype=self.kv_cache_dtype,
        )
        return (input_tokens, input_positions, attn_metadata,
                sampling_metadata, lora_requests, lora_mapping,
                multi_modal_input)

    @torch.inference_mode()
    def profile_run(self) -> None:
        """costa: set `torch.cuda.synchronize(self.device_config.device)`"""
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        # This represents the maximum number of different requests
        # that will have unique loras, an therefore the max amount of memory
        # consumption create dummy lora request copies from the lora request
        # passed in, which contains a lora from the lora warmup path.
        dummy_lora_requests = []
        dummy_lora_requests_per_seq = []
        if self.lora_config:
            for idx in range(self.lora_config.max_loras):
                lora_id = idx + 1
                dummy_lora_request = LoRARequest(
                    lora_name=f"warmup_{lora_id}",
                    lora_int_id=lora_id,
                    lora_local_path="/not/a/real/path",
                )
                self.lora_manager.add_dummy_lora(dummy_lora_request,
                                                 rank=LORA_WARMUP_RANK)
                dummy_lora_requests.append(dummy_lora_request)
            dummy_lora_requests_per_seq = [
                dummy_lora_requests[idx % len(dummy_lora_requests)]
                for idx in range(max_num_seqs)
            ]

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        # Additional GPU memory may be needed for vision encoding, which needs
        # to be accounted for when calculating the GPU blocks for
        # vLLM blocker manager.
        # To exercise the worst scenario for GPU memory consumption,
        # the number of seqs (batch_size) is chosen to maximize the number
        # of images processed.
        if self.vision_language_config:
            max_num_seqs = min(
                max_num_seqs,
                int(max_num_batched_tokens /
                    self.vision_language_config.image_feature_size))
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data, fake_multi_modal_input = _prepare_fake_inputs(
                seq_len, self.vision_language_config)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=dummy_lora_requests_per_seq[group_id]
                if dummy_lora_requests_per_seq else None,
                multi_modal_data=fake_multi_modal_input,
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers
        self.execute_model(seqs, kv_caches)
        torch.cuda.synchronize(self.device_config.device)
        return


class SingleGPUWorker(Worker):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        vision_language_config: Optional[VisionLanguageConfig] = None,
        is_driver_worker: bool = False,
    ) -> None:
        """costa: don't require the driver worker to have rank 0"""
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.load_config = load_config
        self.is_driver_worker = True  # is_driver_worker

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()
        self.vision_language_config = vision_language_config
        if self.vision_language_config:
            assert not self.lora_config, (
                "To be tested: vision language model with LoRA settings.")

        self.model_runner = SingleGPUModelRunner(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            load_config=load_config,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            vision_language_config=vision_language_config,
        )
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: CacheEngine
        self.gpu_cache: List[torch.Tensor]

    def init_device(self) -> None:
        """costa: remove `init_worker_distributed_environment`"""
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            # torch.cuda.set_device(self.device)

            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info(self.device)[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config,
                                        self.device_config)
        self.gpu_cache = self.cache_engine.gpu_cache
        self.model_runner.set_block_size(self.cache_engine.block_size)

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks.

        This also warms up the model, which may record CUDA graphs.
        """
        raise_if_cache_size_invalid(num_gpu_blocks,
                                    self.cache_config.block_size,
                                    self.model_config.max_model_len)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        with torch.cuda.device(self.device_config.device):
            self._init_cache_engine()
            self._warm_up_model()

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None,
        blocks_to_swap_in: Optional[Dict[int, int]] = None,
        blocks_to_swap_out: Optional[Dict[int, int]] = None,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,
        num_lookahead_slots: int = 0,
    ) -> List[SamplerOutput]:

        assert seq_group_metadata_list is not None
        num_seq_groups = len(seq_group_metadata_list)
        assert blocks_to_swap_in is not None
        assert blocks_to_swap_out is not None
        assert blocks_to_copy is not None
        # data: Dict[str, Any] = {
        #     "num_seq_groups": num_seq_groups,
        #     "blocks_to_swap_in": blocks_to_swap_in,
        #     "blocks_to_swap_out": blocks_to_swap_out,
        #     "blocks_to_copy": blocks_to_copy,
        # }

        assert blocks_to_swap_in is not None
        assert blocks_to_swap_out is not None
        assert blocks_to_copy is not None
        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return []

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.gpu_cache)

        # Worker only supports single-step execution. Wrap the output in a list
        # to conform to interface.
        return [output]

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """set `torch.cuda.synchronize(self.device_config.device)`"""
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize(device=self.device_config.device)
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info(
            self.device_config.device)
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        peak_memory = self.init_gpu_memory - free_gpu_memory
        assert peak_memory > 0, (
            "Error in memory profiling. This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")
        cache_block_size = self.get_cache_block_size_bytes()
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization -
             peak_memory) // cache_block_size)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks


class SingleGPUGPUExecutor(GPUExecutor):

    def _init_non_spec_worker(self):
        """costa: don't use `distributed_init_method` and use proper device"""
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker

        assert self.parallel_config.world_size == 1, (
            "GPUExecutor only supports single GPU.")

        distributed_init_method = "dummy"
        self.driver_worker = SingleGPUWorker(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            load_config=self.load_config,
            local_rank=self.device_config.device.index,
            rank=self.device_config.device.index,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            vision_language_config=self.vision_language_config,
            is_driver_worker=True,
        )
        self.driver_worker.init_device()
        self.driver_worker.load_model()


class SingleGPULLMEngine(LLMEngine):

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> "LLMEngine":
        """Use SingleGPUGPUExecutor instead of GPUExecutor."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()

        # Create the LLM engine.
        engine = cls(
            **engine_config.to_dict(),
            executor_class=SingleGPUGPUExecutor,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
        )
        return engine


class SingleGPULLM(LLM):

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )
        self.llm_engine = SingleGPULLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS)
        self.request_counter = Counter()
