import torch
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class WhisperModel:
    def __init__(self, name: str, path: str, audio_length_ms: int = 30000, batch_size: int = 4):
        self.name = name
        self.processor = WhisperProcessor.from_pretrained(path)
        self.model = WhisperForConditionalGeneration.from_pretrained(path)
        self.model.eval()
        self.audio_length_ms = audio_length_ms
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def generate_batch(self, audio_inputs: list) -> list:
        inputs = self.processor(
            audio_inputs,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_features"],
                max_new_tokens=128
            )

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def warmup(self):
        dummy_audio = np.random.randn(16000 * 3).astype(np.float32)
        self.generate_batch([dummy_audio])