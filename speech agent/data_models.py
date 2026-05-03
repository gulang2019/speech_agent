"""Data models for the speech pipeline simulation."""

from dataclasses import dataclass, field
from typing import Optional, Any
import asyncio


@dataclass
class VoiceMetadata:
    conv_id: str
    len_human_frames: int
    len_human_speech: int
    len_gpt_speech: int
    len_gpt_frames: int

    @property
    def show(self) -> str:
        return (
            f"Conv ID: {self.conv_id}, "
            f"Human Frames: {self.len_human_frames}, "
            f"GPT Speech Length: {self.len_gpt_speech}, "
            f"GPT Frames: {self.len_gpt_frames}\n"
        )


@dataclass
class UserData:
    user_index: int = None
    convs: list[VoiceMetadata] = None
    conv_latencies: list[float] = field(default_factory=list)
    current_active_request: Any = None
    current_conv_idx: int = 0