from mini_vllm.offline_inference import OfflineLLM

llm = OfflineLLM(
    model_name = 'facebook/opt-125m'
)

response = llm.generate("Carnegie Mellon Univeristy is known for ", max_tokens = 100, ignore_eos = True)

print(response)