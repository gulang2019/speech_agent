import asyncio
import json
import logging
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# ---- import your Engine ----
# Or paste your Engine class above this file.
from async_stt_engine import Engine

logger = logging.getLogger("ws-asr")
logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

ENGINE: Optional["Engine"] = None
ENGINE_TASK: Optional[asyncio.Task] = None


@app.on_event("startup")
async def startup():
    global ENGINE, ENGINE_TASK
    # Start your Engine once
    ENGINE = Engine(model_name="kyutai/stt-2.6b-en", device="cuda", batch_size=32)
    ENGINE_TASK = asyncio.create_task(ENGINE.step())
    logger.info("Engine started.")


@app.on_event("shutdown")
async def shutdown():
    global ENGINE_TASK
    if ENGINE_TASK:
        ENGINE_TASK.cancel()
        try:
            await ENGINE_TASK
        except asyncio.CancelledError:
            pass


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


async def drain_outputs(ws: WebSocket, out_q: asyncio.Queue):
    """
    Drain tokens from engine out_q and forward to client as text messages.
    """
    while True:
        msg = await out_q.get()
        logger.info(f'sending back {msg}')
        # send tokens/eoa as text frames
        await ws.send_text(str(msg))


@app.websocket("/ws")
async def ws_asr(ws: WebSocket):
    """
    Protocol:
      - Client sends binary messages: Float32Array of shape [frame_size] (mono PCM)
      - Client sends text messages:
          "<eos>" to end utterance (engine will emit "<eoa>" later)
          optional JSON control messages (see below)
      - Server sends text tokens (stream), including "<eoa>"
    """
    assert ENGINE is not None

    await ws.accept()

    # Create one engine request for this websocket session
    in_q, out_q = await ENGINE.add_request(req_id=f"ws-{id(ws)}")

    # Start a task to stream outputs back to client
    out_task = asyncio.create_task(drain_outputs(ws, out_q))

    # Send handshake info (engine framing / sample rate)
    # Client uses this to chunk correctly.
    hello = {
        "type": "hello",
        "sample_rate": int(ENGINE.mimi.sample_rate),
        "frame_rate": float(ENGINE.mimi.frame_rate),
        "frame_size": int(ENGINE.frame_size),
    }
    await ws.send_text(json.dumps(hello))

    try:
        while True:
            msg = await ws.receive()
            
            # logger.info(f'recv {msg}')

            if "text" in msg and msg["text"] is not None:
                text = msg["text"]
                if text == "<eos>":
                    in_q.put_nowait("<eos>")
                elif text == "<eoa>":
                    # usually server-side only, but allow if you want
                    in_q.put_nowait("<eoa>")
                else:
                    # Optional: allow JSON control messages
                    # e.g. {"type":"eos"} or {"type":"reset"}
                    try:
                        obj = json.loads(text)
                        if isinstance(obj, dict) and obj.get("type") == "eos":
                            in_q.put_nowait("<eos>")
                    except Exception:
                        pass

            elif "bytes" in msg and msg["bytes"] is not None:
                b = msg["bytes"]
                # Interpret bytes as float32 PCM samples
                pcm = np.frombuffer(b, dtype=np.float32)

                # Expect exactly frame_size samples per message
                if pcm.shape[0] != ENGINE.frame_size:
                    # You can choose to buffer/rechunk server-side;
                    # for the smoke demo, just reject.
                    logger.warning("Bad frame length: got %d expected %d", pcm.shape[0], ENGINE.frame_size)
                    continue

                t = torch.from_numpy(pcm).to(dtype=torch.float32)
                t = t.view(1, 1, -1)

                # If your engine expects CUDA tensors, either:
                #   - move here:
                # t = t.to(ENGINE.device)
                #   - or move in Engine.prepare_input (recommended)
                in_q.put_nowait(t)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
    finally:
        # try to end gracefully
        try:
            in_q.put_nowait("<eos>")
        except Exception:
            pass

        out_task.cancel()
        try:
            await out_task
        except asyncio.CancelledError:
            pass
