# RoboDojo Xiaomi bridge

This repository can serve the existing single-arm Xiaomi RoboCasa checkpoint
to RoboDojo's `pour_liquid_into_cup` task through the public XPolicyLab
WebSocket protocol.

The bridge is compatibility-only. RoboDojo's official Xiaomi adapter uses a
RoboDojo-trained checkpoint and a dual-arm state/action contract. The local
RoboCasa checkpoint instead consumes one 7-joint-plus-gripper state and emits
7D RoboCasa actions. The bridge controls one RoboDojo arm, sends the other arm
an explicit hold target, and converts the action to an absolute `ee_pose` plus a
normalized gripper target.

## Run

RoboDojo and XPolicyLab must be installed in their own environments. Put the
XPolicyLab checkout on `PYTHONPATH` when starting the model server:

```bash
PYTHONPATH=/path/to/XPolicyLab:$PYTHONPATH \
  python scripts/robodojo_xiaomi_server.py \
  --model-path models/xiaomi-robotics-1-robocasa \
  --port 19000
```

Configure RoboDojo's evaluation config with:

```text
task_name: pour_liquid_into_cup
config_name: pour_liquid_into_cup
env_cfg_type: arx_x5
eval_batch: false
```

Configure its deployment config with:

```text
policy_name: Xiaomi_Robotics_1
protocol: ws
host: 127.0.0.1
port: 19000
```

Use the existing `XPolicyLab.policy.Xiaomi_Robotics_1.deploy` module for the
single-environment loop, or copy that `deploy.py` under a local policy name.
The bridge implements its `reset`, `update_obs`, and `get_action` calls; no
checkpoint-specific changes are required in RoboDojo.

The default bridge settings are `active_arm=right`, `action_mode=delta_ee`,
`delta_frame=base`, `action_units=robocasa_normalized`, and
`gripper_mode=auto`. The normalized action is converted using the RoboCasa
OSC-Pose scales of `0.05 m` and `0.5 rad`; the ARX X5 base rotation is
configurable in the Python adapter. These are command-line options and must
remain explicit when evaluating a checkpoint with different action semantics.

## Compatibility policy

The bridge accepts RoboDojo observation format `v1.0` only. Camera names are
resolved by aliases rather than list position, and missing cameras or unknown
format versions fail closed. This is intentional: a future RoboDojo schema
must add a versioned adapter instead of silently changing the action frame.
For the current dual-arm `arx_x5` contract, every returned EE action contains
both arm pose/gripper keys; the non-active arm receives its current state as a
hold target.

The exact task metadata is in
[`configs/robodojo_tasks.json`](../configs/robodojo_tasks.json), and the
conversion code is in
[`scripts/robodojo_compat.py`](../scripts/robodojo_compat.py).

## One-click deployment

From a fresh checkout, the deployment script downloads missing user-local
resources and then runs the preflight and delay x replan smoke:

```bash
bash scripts/robodojo_pour_liquid_one_click.sh
```

The default mode submits one GPU job to `gpu-scavenger`. For a process that
already has a GPU allocation, run directly with an explicit visibility mask:

```bash
CUDA_VISIBLE_DEVICES=0 ROBODOJO_RUN_MODE=direct \
  bash scripts/robodojo_pour_liquid_one_click.sh
```

The script requires host-provided `nvidia-smi`, Bubblewrap, Singularity,
`unsquashfs`, `curl`, and Python >=3.10. Slurm mode additionally requires
`sbatch`. It does not install system packages or modify global GPU/runtime
configuration. Everything it downloads or builds is placed below `.deps/`,
`models/`, or `.runtime/`.
