"""Diagnose door-task progress: log fixture type and per-door joint states."""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
sys.path.insert(0, "/fact_home/xunyuanliu/dev/robo")
from scripts.xiaomi_latency_experiment import (
    PolicyClient, render_observation, make_action, CAMERA_NAMES, resolve_task,
)
from robocasa.utils.env_utils import create_env
from robocasa.utils.dataset_registry_utils import get_task_horizon

task = sys.argv[1]
episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 2
SEG = ((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))
env_name, alias_kwargs, alias_horizon = resolve_task(task)
h = int(sys.argv[3]) if len(sys.argv) > 3 else (alias_horizon or get_task_horizon(task))
replan = int(sys.argv[4]) if len(sys.argv) > 4 else 10
print("env_name", env_name, "kwargs", alias_kwargs, "horizon", h, flush=True)
policy = PolicyClient("/fact_home/xunyuanliu/dev/robo/models/xiaomi-robotics-1-robocasa", 42, 5, "eager")

for ep in range(episodes):
    env = create_env(env_name=env_name, robots="PandaOmron", camera_names=list(CAMERA_NAMES),
                     camera_widths=256, camera_heights=256, seed=ep, render_onscreen=False,
                     randomize_cameras=False, split=None, obj_instance_split="pretrain",
                     layout_and_style_ids=SEG, **alias_kwargs)
    env.reset()
    fxtr = env.fxtr
    print(f"--- {task} ep{ep} layout={env.layout_id} style={env.style_id} fixture={type(fxtr).__name__}")
    print("    joints:", fxtr.door_joint_names)
    images, state = render_observation(env, 256, 0.95)
    plan = list(policy.infer(state, images, env.get_ep_meta()["lang"]))
    next_req = replan
    for step in range(h):
        if step >= next_req:
            images, state = render_observation(env, 256, 0.95)
            plan = list(policy.infer(state, images, env.get_ep_meta()["lang"]))
            next_req = step + replan
        if step % 100 == 0:
            st = fxtr.get_joint_state(env, fxtr.door_joint_names)
            print("   step", step, {k: round(float(v), 3) for k, v in st.items()},
                  "is_open=", fxtr.is_open(env), flush=True)
        env.step(make_action(env, plan.pop(0)))
        if env._check_success():
            st = fxtr.get_joint_state(env, fxtr.door_joint_names)
            print("  SUCCESS step", step, {k: round(float(v), 3) for k, v in st.items()}, flush=True)
            break
    print("  final success=", env._check_success(), flush=True)
    env.close()
policy.close()
