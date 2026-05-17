from mini_vllm.offline_inference import OfflineLLM

llm = OfflineLLM(
    model_name = 'facebook/opt-1.3b'
)

# llm = OfflineLLM(
#     model_name = 'facebook/opt-2.7b'
# )

# llm = OfflineLLM(
#     model_name = 'facebook/opt-6.7b'
# )

# llm = OfflineLLM(
#     model_name = 'facebook/opt-13b'
# )

# llm = OfflineLLM(
#     model_name = 'meta-llama/Llama-2-7b'
# )

response = llm.generate("Carnegie Mellon Univeristy is known for ", max_tokens = 100, ignore_eos = True)

print(response)