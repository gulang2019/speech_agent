"""Diagnose why composite tasks score 0: log subgoal progress over an episode."""
import os, sys, time, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
sys.path.insert(0, "/fact_home/xunyuanliu/dev/robo")
from scripts.xiaomi_latency_experiment import (
    PolicyClient, render_observation, make_action, CAMERA_NAMES,
)
from robocasa.utils.env_utils import create_env
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils import object_utils as OU

task = sys.argv[1]
episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 2
SEG = ((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))
h = get_task_horizon(task)
policy = PolicyClient("/fact_home/xunyuanliu/dev/robo/models/xiaomi-robotics-1-robocasa", 42, 5, "eager")

for ep in range(episodes):
    seed = ep
    env = create_env(env_name=task, robots="PandaOmron", camera_names=list(CAMERA_NAMES),
                     camera_widths=256, camera_heights=256, seed=seed, render_onscreen=False,
                     randomize_cameras=False, split=None, obj_instance_split="pretrain",
                     layout_and_style_ids=SEG)
    env.reset()
    mw = env.get_fixture("microwave") if "microwave" in [f.name for f in env.fixtures.values()] else None
    print(f"--- {task} ep{ep} seed{seed} layout={env.layout_id} style={env.style_id} horizon={h}", flush=True)
    images, state = render_observation(env, 256, 0.95)
    plan = list(policy.infer(state, images, env.get_ep_meta()["lang"]))
    trace = []
    for step in range(h):
        if step % 5 == 0:
            try:
                obj_pos = env.sim.data.body_xpos[env.obj_body_id["obj"]].copy()
                grip = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]].copy()
                rec = dict(step=step, obj=list(np.round(obj_pos,3)), grip=list(np.round(grip,3)),
                           grip_obj_dist=float(np.linalg.norm(grip-obj_pos)),
                           obj_far=bool(OU.gripper_obj_far(env, th=0.25)))
                if mw is not None:
                    rec["mw_door_open"] = bool(mw.is_open(env))
                    rec["obj_in_mw"] = bool(OU.obj_inside_of(env, "obj", mw))
                trace.append(rec)
            except Exception as e:
                trace.append(dict(step=step, err=str(e)))
        if not plan:
            images, state = render_observation(env, 256, 0.95)
            plan = list(policy.infer(state, images, env.get_ep_meta()["lang"]))
        env.step(make_action(env, plan.pop(0)))
        if env._check_success():
            print(f"  SUCCESS at step {step}", flush=True)
            break
    print("  final_success=", env._check_success(), flush=True)
    for rec in trace[::4][:14]:
        print("   ", json.dumps(rec), flush=True)
    env.close()
policy.close()
