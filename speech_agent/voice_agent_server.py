import asyncio
import json
import logging
import os
from pathlib import Path
import threading
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from async_stt_engine import STTEngine
from async_tts_engine import TTSEngine

logger = logging.getLogger("voice-agent")
logging.basicConfig(level=logging.INFO)


@dataclass(frozen=True)
class VLLMConfig:
    base_url: str = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    model: str = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    temperature: float = float(os.environ.get("VLLM_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.environ.get("VLLM_MAX_TOKENS", "256"))
    timeout_s: float = float(os.environ.get("VLLM_TIMEOUT_S", "60"))
    stream: bool = os.environ.get("VLLM_STREAM", "1") not in ("0", "false", "False")
    api_key: str = os.environ.get("VLLM_API_KEY", "tok-123")
    system_prompt: str = os.environ.get(
        "VLLM_SYSTEM_PROMPT",
        "You are a concise, helpful voice assistant. Keep answers short.",
    )

def _build_headers(api_key: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _post_json(url: str, payload: dict, timeout_s: float, headers: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _post_json_stream(url: str, payload: dict, timeout_s: float, headers: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            yield json.loads(data_str)


def _extract_llm_text(resp: dict) -> str:
    try:
        choice = resp["choices"][0]
        if "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]
        if "text" in choice:
            return choice["text"]
    except Exception:
        pass
    return ""


def _extract_llm_delta(resp: dict) -> str:
    try:
        choice = resp["choices"][0]
        delta = choice.get("delta", {})
        if isinstance(delta, dict):
            text = delta.get("content", "")
            if text:
                return text
        if "text" in choice:
            return choice["text"]
    except Exception:
        pass
    return ""


async def vllm_chat(cfg: VLLMConfig, messages: list[dict]) -> str:
    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": False,
    }
    headers = _build_headers(cfg.api_key)
    resp = await asyncio.to_thread(_post_json, url, payload, cfg.timeout_s, headers)
    return _extract_llm_text(resp).strip()


async def vllm_chat_stream(cfg: VLLMConfig, messages: list[dict]):
    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": True,
    }
    headers = _build_headers(cfg.api_key)
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[object] = asyncio.Queue()
    done_sentinel = object()

    def _run():
        try:
            for item in _post_json_stream(url, payload, cfg.timeout_s, headers):
                delta = _extract_llm_delta(item)
                if delta:
                    asyncio.run_coroutine_threadsafe(q.put(delta), loop)
            asyncio.run_coroutine_threadsafe(q.put(done_sentinel), loop)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(q.put(exc), loop)

    threading.Thread(target=_run, daemon=True).start()

    while True:
        item = await q.get()
        if item is done_sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield str(item)


class VoiceSession:
    def __init__(
        self,
        ws: WebSocket,
        stt_engine: STTEngine,
        tts_engine: TTSEngine,
        llm_cfg: VLLMConfig,
    ):
        self.ws = ws
        self.stt_engine = stt_engine
        self.tts_engine = tts_engine
        self.llm_cfg = llm_cfg
        self.messages = []
        if llm_cfg.system_prompt:
            self.messages.append({"role": "system", "content": llm_cfg.system_prompt})
        self.user_text_q: asyncio.Queue[str] = asyncio.Queue()
        self._drain_stt_task: Optional[asyncio.Task] = None
        self._drain_tts_task: Optional[asyncio.Task] = None 
        self._llm_task: Optional[asyncio.Task] = None
        self.stt_in: Optional[asyncio.Queue] = None
        self.stt_out: Optional[asyncio.Queue] = None
        self.tts_in: Optional[asyncio.Queue] = None
        self.tts_out: Optional[asyncio.Queue] = None
        
        logger.info(f'using llm config: {llm_cfg}.')

    async def start(self):
        self._llm_task = asyncio.create_task(self._llm_worker())

    def start_stt(self):
        self.stt_in, self.stt_out = self.stt_engine.add_request(
            req_id=f"stt-{id(self.ws)}"
        )
        self._drain_stt_task = asyncio.create_task(self._drain_stt())

    def start_tts(self):
        self.tts_in, self.tts_out = self.tts_engine.add_request(
            req_id=f"tts-{id(self.ws)}"
        )
        self._drain_tts_task = asyncio.create_task(self._drain_tts())

    async def close(self):
        for task in [self._llm_task, self._drain_stt_task, self._drain_tts_task]:
            task.cancel()
        for task in [self._llm_task, self._drain_stt_task, self._drain_tts_task]:
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def send_json(self, obj: dict):
        await self.ws.send_text(json.dumps(obj))

    async def send_status(self, state: str):
        await self.send_json({"type": "status", "state": state})

    async def enqueue_user_text(self, text: str):
        text = text.strip()
        if not text:
            return
        await self.send_json({"type": "stt_final", "text": text})
        await self.user_text_q.put(text)

    def reset(self):
        self.messages = []
        if self.llm_cfg.system_prompt:
            self.messages.append({"role": "system", "content": self.llm_cfg.system_prompt})

    async def _drain_stt(self):
        assert self.stt_out is not None
        stt_buf = []
        while True:
            msg = await self.stt_out.get()
            if msg == "<eoa>":
                text = "".join(stt_buf).strip()
                if text:
                    await self.send_json({"type": "stt_final", "text": text})
                    await self.user_text_q.put(text)
                break
            else:
                stt_buf.append(str(msg))
                await self.send_json({"type": "stt", "text": str(msg)})

    async def _drain_tts(self):
        assert self.tts_out is not None
        while True:
            pcm = await self.tts_out.get()
            if pcm is None:
                await self.send_json({"type": "tts_end"})
                await self.send_status("idle")
                self.tts_in = None
                self.tts_out = None
                break
            if isinstance(pcm, np.ndarray):
                data = pcm.astype(np.float32, copy=False).tobytes()
            else:
                data = np.asarray(pcm, dtype=np.float32).tobytes()
            await self.ws.send_bytes(data)

    async def _llm_worker(self):
        while True:
            user_text = await self.user_text_q.get()
            self.start_tts()
            await self.send_status("thinking")
            self.messages.append({"role": "user", "content": user_text})
            try:
                if self.llm_cfg.stream:
                    reply = await self._stream_llm_reply()
                else:
                    reply = await vllm_chat(self.llm_cfg, self.messages)
            except Exception as exc:
                logger.exception("LLM call failed: %s", exc)
                await self.send_json({"type": "error", "message": "LLM request failed"})
                continue
            if not reply:
                await self.send_json({"type": "error", "message": "Empty LLM response"})
                continue
            self.messages.append({"role": "assistant", "content": reply})
            await self.send_json({"type": "llm", "text": reply})
            if not self.llm_cfg.stream:
                await self.send_status("speaking")
                for word in reply.split():
                    self.tts_in.put_nowait(word)
                self.tts_in.put_nowait("<eos>")

    async def _stream_llm_reply(self) -> str:
        assert self.tts_in is not None
        reply_parts: list[str] = []
        tts_buf = ""
        started_speaking = False
        chunk_punct = (".", "!", "?", ";", ":")
        async for delta in vllm_chat_stream(self.llm_cfg, self.messages):
            reply_parts.append(delta)
            await self.send_json({"type": "llm_delta", "text": delta})
            tts_buf += delta
            chunks: list[str] = []
            if any(ch.isspace() for ch in tts_buf):
                if tts_buf[-1].isspace():
                    chunks = tts_buf.split()
                    tts_buf = ""
                else:
                    parts = tts_buf.split()
                    tts_buf = parts.pop() if parts else ""
                    chunks = parts
            elif tts_buf.endswith(chunk_punct) or len(tts_buf) >= 24:
                chunks = [tts_buf]
                tts_buf = ""
            if chunks:
                if not started_speaking:
                    started_speaking = True
                    await self.send_status("speaking")
                for chunk in chunks:
                    self.tts_in.put_nowait(chunk)
        reply = "".join(reply_parts).strip()
        if tts_buf:
            if not started_speaking:
                started_speaking = True
                await self.send_status("speaking")
            self.tts_in.put_nowait(tts_buf)
        if started_speaking:
            self.tts_in.put_nowait("<eos>")
        return reply


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

STT_ENGINE: Optional[STTEngine] = None
TTS_ENGINE: Optional[TTSEngine] = None
STT_TASK: Optional[asyncio.Task] = None
TTS_TASK: Optional[asyncio.Task] = None
LLM_CFG = VLLMConfig()


@app.on_event("startup")
async def startup():
    global STT_ENGINE, TTS_ENGINE, STT_TASK, TTS_TASK
    STT_ENGINE = STTEngine(model_name="kyutai/stt-2.6b-en", device="cuda", batch_size=32)
    TTS_ENGINE = TTSEngine(device="cuda", batch_size=8)
    STT_TASK = asyncio.create_task(STT_ENGINE.run())
    TTS_TASK = asyncio.create_task(TTS_ENGINE.run())
    logger.info("STT/TTS engines started.")


@app.on_event("shutdown")
async def shutdown():
    global STT_TASK, TTS_TASK
    for task in (STT_TASK, TTS_TASK):
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(STATIC_DIR / "voice_agent.html", "r", encoding="utf-8") as f:
        return f.read()


@app.websocket("/ws")
async def ws_voice(ws: WebSocket):
    assert STT_ENGINE is not None
    assert TTS_ENGINE is not None
    await ws.accept()

    session = VoiceSession(ws, STT_ENGINE, TTS_ENGINE, LLM_CFG)
    await session.start()

    hello = {
        "type": "hello",
        "input_sample_rate": int(STT_ENGINE.mimi.sample_rate),
        "input_frame_size": int(STT_ENGINE.frame_size),
        "output_sample_rate": int(TTS_ENGINE.sample_rate),
        "output_frame_size": int(TTS_ENGINE.frame_size),
        "llm_model": LLM_CFG.model,
    }
    await session.send_json(hello)
    await session.send_status("idle")

    try:
        while True:
            msg = await ws.receive()
            if "text" in msg and msg["text"] is not None:
                text = msg["text"]
                if text == "<eos>":
                    if session.stt_in is not None:
                        session.stt_in.put_nowait("<eos>")
                    continue
                try:
                    obj = json.loads(text)
                except Exception:
                    obj = None
                if isinstance(obj, dict):
                    msg_type = obj.get("type")
                    if msg_type == "eos":
                        if session.stt_in is not None:
                            session.stt_in.put_nowait("<eos>")
                    elif msg_type == "start_mic":
                        session.start_stt()
                    elif msg_type == "user_text":
                        await session.enqueue_user_text(obj.get("text", ""))
                    elif msg_type == "reset":
                        session.reset()
                        await session.send_status("idle")
                else:
                    await session.enqueue_user_text(text)
            elif "bytes" in msg and msg["bytes"] is not None:
                pcm = np.frombuffer(msg["bytes"], dtype=np.float32)
                if session.stt_in is None:
                    logger.warning("Dropping audio before start_mic.")
                    continue
                if pcm.shape[0] != STT_ENGINE.frame_size:
                    logger.warning(
                        "Bad frame length: got %d expected %d",
                        pcm.shape[0],
                        STT_ENGINE.frame_size,
                    )
                    continue
                t = torch.from_numpy(pcm).to(dtype=torch.float32).view(1, 1, -1)
                session.stt_in.put_nowait(t)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as exc:
        logger.exception("WebSocket error: %s", exc)
    finally:
        try:
            if session.stt_in is not None:
                session.stt_in.put_nowait("<eos>")
        except Exception:
            pass
        await session.close()
