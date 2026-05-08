import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM


class LLMModel:
    def __init__(self, name: str, path: str, max_tokens: int = 256, batch_size: int = 8):
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
        self.model.eval()
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def generate_batch(self, prompts: list) -> list:
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False
            )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def warmup(self):
        self.generate_batch(["warmup"])