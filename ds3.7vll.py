import time

import torch
from accelerate import Accelerator
from accelerate.state import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm import SamplingParams, SingleGPULLM

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
prompt_ids = tok.batch_encode_plus(prompts)["input_ids"]
accelerator = Accelerator(gradient_accumulation_steps=2)
state = PartialState()
llm2 = AutoModelForCausalLM.from_pretrained(
    "gradientai/Llama-3-8B-Instruct-262k")
llm2 = llm2.to(accelerator.device)
accelerator.print(f"{torch.cuda.device_count()=}")
if state.is_main_process:
    sampling_params = SamplingParams(temperature=0.001, top_p=1.0)
    llm = SingleGPULLM(model="meta-llama/Meta-Llama-3-8B",
                       tensor_parallel_size=1,
                       device="cuda:7")
    llmp = llm.llm_engine.model_executor.driver_worker.model_runner.model
    print(f"🔥🔥🔥 vllm lives in {llmp.lm_head.weight.device}")
    print("prepare to generate")
    outputs = llm.generate(prompt_token_ids=prompt_ids,
                           sampling_params=sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    print("🔥🔥🔥 Loading weights using shared memory;"
          "we expect the generations to be completely different")
    start_time = time.time()
    llmp.load_weights(llm2.named_parameters())
    print(f"Time to load weights: {time.time() - start_time:.2f} seconds")
    outputs = llm.generate(prompt_token_ids=prompt_ids,
                           sampling_params=sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
else:
    time.sleep(1000)
    print("I'm waiting for the main process to generate...")
accelerator.wait_for_everyone()
