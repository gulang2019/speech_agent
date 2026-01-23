from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import argparse
import asyncio
import os

from .engine import Engine 
from .struct import Config
from .vllm_utils import get_engine_from_vllm

_engine: Engine = None 
_engine_task: asyncio.Task = None 

app = FastAPI()

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int | None = None
    ignore_eos: bool = False

class CompletionResponse(BaseModel):
    text: str


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_memory_utilization", type=float, default=0.8)
    return parser.parse_args()


def _args_from_env():
    model_name = os.environ.get("MINI_VLLM_MODEL_NAME")
    if not model_name:
        return None
    max_memory = float(os.environ.get("MINI_VLLM_MAX_MEMORY_UTILIZATION", "0.8"))
    return argparse.Namespace(
        model_name=model_name,
        max_memory_utilization=max_memory,
    )


def init(args):
    config = Config(
        model_name = args.model_name, 
        max_memory_utilization= args.max_memory_utilization
    )
    global _engine
    _engine = get_engine_from_vllm(config)
    return _engine


@app.on_event("startup")
async def _startup():
    global _engine, _engine_task
    if _engine is None:
        args = _args_from_env()
        if args is None:
            raise RuntimeError(
                "Model not configured. Set MINI_VLLM_MODEL_NAME or run as a script with --model_name."
            )
        _engine = init(args)
    if _engine_task is None:
        _engine_task = asyncio.create_task(_engine.run())
    if args is None:
        raise RuntimeError(
            "Model not configured. Set MINI_VLLM_MODEL_NAME or run as a script with --model_name."
        )
        
@app.post('/completion')
async def completion(request: CompletionRequest) -> CompletionResponse:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")
    token_gen = await _engine.add_request(request.dict())
    tokens = []
    async for tok in token_gen():
        tokens.append(tok)
    return CompletionResponse(text="".join(tokens))


if __name__ == "__main__":
    args = _parse_args()
    os.environ["MINI_VLLM_MODEL_NAME"] = args.model_name
    os.environ["MINI_VLLM_MAX_MEMORY_UTILIZATION"] = str(args.max_memory_utilization)
    import uvicorn

    uvicorn.run("mini_vllm.api_server:app", host="0.0.0.0", port=8000)
