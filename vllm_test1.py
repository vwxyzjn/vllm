
from transformers import AutoTokenizer

from vllm import SamplingParams, SingleGPULLM

tok = AutoTokenizer.from_pretrained("vwxyzjn/ppo_zephyr7")
prompts = [
    {"role": "user", "content": "Compose a speech about the need for more affordable dental care."},
]

prompt_ids = tok.apply_chat_template(prompts, add_generation_prompt=True)
sampling_params = SamplingParams(temperature=0.001, top_p=1.0, max_tokens=1024, include_stop_str_in_output=True)
llm = SingleGPULLM(model="vwxyzjn/ppo_zephyr7",
                    tensor_parallel_size=1,
                    device="cuda:7")
llmp = llm.llm_engine.model_executor.driver_worker.model_runner.model
print(f"🔥🔥🔥 vllm lives in {llmp.lm_head.weight.device}")
print("prepare to generate")
outputs = llm.generate(prompt_token_ids=[prompt_ids],
                        sampling_params=sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

