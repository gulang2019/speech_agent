from abc import ABC, abstractmethod
from typing import Any, Dict
import torch
from .models.asr import WhisperModel
from .models.lm import LLMModel, Seq2SeqModel


class ModelLoader(ABC):
    @abstractmethod
    def load(self, name: str, path: str, **kwargs) -> Any:
        pass


class ASRLoader(ModelLoader):
    def load(self, name: str, path: str, **kwargs) -> WhisperModel:
        return WhisperModel(name, path, **kwargs)


class LMLoader(ModelLoader):
    def load(self, name: str, path: str, **kwargs) -> LLMModel:
        return LLMModel(name, path, **kwargs)


class Seq2SeqLoader(ModelLoader):
    def load(self, name: str, path: str, **kwargs) -> Seq2SeqModel:
        return Seq2SeqModel(name, path, **kwargs)


def get_loader(model_type: str) -> ModelLoader:
    loaders = {
        'asr': ASRLoader(),
        'lm': LMLoader(),
        'seq2seq': Seq2SeqLoader(),
    }
    return loaders[model_type]