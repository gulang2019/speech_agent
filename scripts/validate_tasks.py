"""Validate that tasks build and reset on the legacy5 scene set."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import robocasa  # noqa: F401
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import create_env

SCENES = ((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))
CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")

for task in sys.argv[1:]:
    try:
        horizon = get_task_horizon(task)
    except Exception as error:
        print(f"=== {task} horizon=UNKNOWN ({error})", flush=True)
    else:
        print(f"=== {task} horizon={horizon}", flush=True)
    for layout, style in SCENES:
        try:
            env = create_env(
                env_name=task, robots="PandaOmron", camera_names=list(CAMERAS),
                camera_widths=64, camera_heights=64, seed=0, render_onscreen=False,
                randomize_cameras=False, split=None, obj_instance_split="pretrain",
                layout_and_style_ids=((layout, style),),
            )
            env.reset()
            print(f"  OK {layout}/{style}: {env.get_ep_meta()['lang']!r}", flush=True)
            env.close()
        except Exception as error:
            print(f"  FAIL {layout}/{style}: {type(error).__name__}: {error}", flush=True)
