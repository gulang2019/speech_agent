import os 
import json 
import urllib 
from dataclasses import dataclass
import asyncio
import threading

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


async def vllm_chat_stream(cfg: VLLMConfig, messages: list[dict], **kwargs):
    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": True,
    }
    payload.update(kwargs)
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
