"""Asynchronous RoboDojo deployment loop for delay x replan smoke sweeps.

The stock RoboDojo deployment loop waits for every inference before advancing
the simulator. That makes an injected sleep a wall-clock pause rather than a
simulated stale-action delay. This loop keeps executing the current action
chunk while the next ``update_obs``/``get_action`` request is in flight.

The simulator-side contract remains the official XPolicyLab contract. The
experiment controls are read from ``ROBODOJO_*`` environment variables. The
first inference is a bootstrap request and is not delayed.
"""

from __future__ import annotations

import copy
import json
import os
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {parsed}")
    return parsed


def _append_jsonl(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _request(model_client: Any, obs: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep all calls on one worker thread; WsModelClient is not thread-safe."""

    model_client.call(func_name="update_obs", obs=obs)
    return model_client.call(func_name="get_action")


def _delay_steps() -> int:
    delay_ms = _env_int("ROBODOJO_DELAY_MS", 0)
    frequency = float(os.environ.get("ROBODOJO_DELAY_STEP_FREQUENCY", "25"))
    if delay_ms == 0:
        return 0
    if frequency <= 0:
        raise ValueError("ROBODOJO_DELAY_STEP_FREQUENCY must be positive")
    return int(round(delay_ms * frequency / 1000.0))


def _run_one_episode(
    task_env: Any, model_client: Any, executor: ThreadPoolExecutor
) -> dict[str, Any]:
    delay_ms = _env_int("ROBODOJO_DELAY_MS", 0)
    delay_steps = _delay_steps()
    replan_steps = _env_int("ROBODOJO_REPLAN_STEPS", 5)
    if replan_steps < 1:
        raise ValueError("ROBODOJO_REPLAN_STEPS must be positive")

    model_client.call(func_name="reset")

    active_actions: deque[dict[str, Any]] = deque()
    pending: Future[list[dict[str, Any]]] | None = None
    pending_available_step = 0
    next_request_step = 0
    step = 0
    inference_requests = 0
    replaced_actions = 0
    inference_wait_s = 0.0
    episode_start = time.perf_counter()

    def launch(request_step: int, *, bootstrap: bool = False) -> None:
        nonlocal pending, pending_available_step, inference_requests
        if pending is not None:
            return
        obs = copy.deepcopy(task_env.get_obs())
        pending_available_step = request_step if bootstrap else request_step + delay_steps
        pending = executor.submit(_request, model_client, obs)
        inference_requests += 1

    def install_if_ready(force: bool = False) -> bool:
        nonlocal pending, active_actions, replaced_actions, inference_wait_s
        if pending is None:
            return False
        if not force and step < pending_available_step:
            return False
        wait_start = time.perf_counter()
        actions = pending.result()
        inference_wait_s += time.perf_counter() - wait_start
        pending = None
        replaced_actions += len(active_actions)
        active_actions = deque(actions)
        return True

    # Bootstrap is intentionally blocking, as in the existing latency runs.
    launch(0, bootstrap=True)
    install_if_ready(force=True)
    next_request_step = replan_steps

    while not task_env.is_episode_end():
        if step >= next_request_step and pending is None:
            launch(step)
            next_request_step += replan_steps

        install_if_ready()
        if not active_actions:
            # Do not synthesize a fallback robot command when a delayed model
            # response is not yet eligible; wait for the real response.
            if pending is None:
                launch(step)
            install_if_ready(force=True)

        if not active_actions:
            raise RuntimeError("policy returned an empty action chunk")
        task_env.take_action(active_actions.popleft())
        step += 1

    if pending is not None:
        pending.result()

    result = {
        "delay_ms": delay_ms,
        "delay_steps": delay_steps,
        "replan_steps": replan_steps,
        "steps": step,
        "inference_requests": inference_requests,
        "queued_actions_replaced": replaced_actions,
        "model_wait_s": inference_wait_s,
        "wall_time_s": time.perf_counter() - episode_start,
    }
    _append_jsonl(os.environ.get("ROBODOJO_SWEEP_METRICS"), result)
    return result


def eval_one_episode(TASK_ENV: Any, model_client: Any) -> None:
    """Entry point consumed by RoboDojo's ``EvalEnv``."""

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="robodojo-policy") as executor:
        _run_one_episode(TASK_ENV, model_client, executor)
