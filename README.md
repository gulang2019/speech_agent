<h1 align="center">RoboCasa</h1>
<!-- ![alt text](https://github.com/UT-Austin-RPL/maple/blob/web/src/overview.png) -->
<img src="docs/images/readme.webp" width="100%" />

**RoboCasa** is a large-scale simulation framework for training generally capable robots to perform everyday tasks. It was [originally released](https://robocasa.ai/assets/robocasa_rss24.pdf) in 2024 by UT Austin researchers. The latest iteration, **RoboCasa365**, builds upon the original release with significant new functionalities to support large-scale training and benchmarking in sim. Four pillars underlie RoboCasa365:
- **Diverse tasks**: 365 tasks created with the guidance of large language models
- **Diverse assets**: including 2,500+ kitchen scenes and 3,200+ 3D objects
- **High-quality demonstrations**: including 600+ hours of human demonstrations in addition to 1,600+ hours of robot datasets created with automated trajectory tools
- **Benchmarking support**: popular policy learning methods including Diffusion Policy, pi, and GR00T, plus user-submitted models on the [leaderboard](https://robocasa.ai/leaderboard.html)


This guide contains information about installation and setup. Please refer to the following resources for additional information:

[**[Home page]**](https://robocasa.ai) &ensp; [**[Documentation]**](https://robocasa.ai/docs/introduction/overview.html) &ensp; [**[RoboCasa365 Paper]**](https://robocasa.ai/assets/robocasa365_iclr26.pdf) &ensp; [**[Original RoboCasa Paper]**](https://robocasa.ai/assets/robocasa_rss24.pdf) &ensp; [**[Leaderboard]**](https://robocasa.ai/leaderboard.html)

-------
## Updates
* [7/7/2026] Our target composite task datasets have been updated to include per-frame **subtask annotations**. Every timestep is labeled with a subtask index, atomic-skill name, stage (i.e. pick / place / navigate), and a natural-language instruction, to support hierarchical policy learning.
* [5/12/2026] **v1.0.1**: Updated horizon lengths (1.5x increase) across all tasks for consistency. Please update to the latest version for running evals.
* [2/18/2026] **v1.0**: RoboCasa365 release, with 365 tasks, 2500+ kitchen scenes, 2200+ hours of robot demonstration data, and benchmarking support.
* [10/31/2024] **v0.2**: using RoboSuite `v1.5` as the backend, with improved support for custom robot composition, composite controllers, more teleoperation devices, photo-realistic rendering.

## Table of Contents
- [Updates](#updates)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Tasks, datasets, policy learning, and additional use cases](#tasks-datasets-policy-learning-and-additional-use-cases)
- [License](#license)
- [Citation](#citation)

-------
## Installation
> **Local setup with Xiaomi-Robotics-1 evaluation and the cross-model
> injected-delay / replan / RTC benchmark: see [INSTALL.md](INSTALL.md).**
> Delays are injected in simulation steps by default (`--delay-domain sim`), so
> the injected delay is the only source of staleness; pass `--delay-domain wall`
> to reproduce the archived wall-clock results in
> `eval_results/archive/legacy_wallclock_delay/`.

This branch also contains the reproducible code for the Xiaomi-Robotics-1 / GR00T
N1.5 latency benchmark. Model weights, local environments, kitchen assets, logs,
and generated evaluation outputs are intentionally not tracked in Git. See
[INSTALL.md](INSTALL.md) for the full machine setup and [the Stage 1 handoff](docs/stage1_xiaomi_gr00t_7task_heatmap.md)
for the seven-task Slurm experiment.

RoboCasa works across all major computing platforms. The easiest way to set up is through the [Anaconda](https://www.anaconda.com/) package management system. Follow the instructions below to install:
1. Set up conda environment:

   ```sh
   conda create -c conda-forge -n robocasa python=3.11
   ```
2. Activate conda environment:
   ```sh
   conda activate robocasa
   ```
3. Clone and setup robosuite dependency (**important: use the master branch!**):

   ```sh
   git clone https://github.com/ARISE-Initiative/robosuite
   cd robosuite
   pip install -e .
   ```
4. Clone and setup this repo:

   ```sh
   cd ..
   git clone https://github.com/robocasa/robocasa
   cd robocasa
   pip install -e .
   pip install pre-commit; pre-commit install           # Optional: set up code formatter.

   (optional: if running into issues with numba/numpy, run: conda install -c numba numba=0.56.4 -y)
   ```
5. Install the package and download assets:
   ```sh
   python -m robocasa.scripts.setup_macros              # Set up system variables.
   python -m robocasa.scripts.download_kitchen_assets   # Caution: Assets to be downloaded are around 10GB.
   ```

-------
## Basic Usage

### Delay benchmark quick start

Run these commands from the repository root after installing the dependencies
and downloading the required local checkpoints:

```bash
cd /path/to/robocasa
.conda-env/bin/python scripts/validate_latency_manifest.py

# One Xiaomi condition, locally
.conda-env/bin/python scripts/xiaomi_latency_experiment.py \
  --model-name xiaomi \
  --task-name PickPlaceCounterToCabinet \
  --scene-set legacy5 --object-split pretrain \
  --delays-ms 0 --delay-domain sim \
  --replan-steps-list 5 --rtc off --episodes 3 \
  --output-dir eval_results/quickstart/xiaomi
```

For a GPU cluster, use the Slurm wrapper. Environment variables override the
defaults in `slurm/xiaomi_latency_experiment.sbatch`:

```bash
MODEL_NAME=xiaomi \
TASK_NAME=PickPlaceCounterToSink \
SCENE_SET=pretrain20 OBJECT_SPLIT=pretrain \
DELAYS_MS=0,100,300,500 DELAY_DOMAIN=sim \
REPLAN_STEPS_LIST=1,2,3,4,5,6,8,10,12,15,20 \
RTC=off CONTROL_FREQUENCY=20 EPISODES=3 \
OUTPUT_DIR=eval_results/stage1/xiaomi/PickPlaceCounterToSink \
  sbatch slurm/xiaomi_latency_experiment.sbatch
```

Set `MODEL_NAME=gr00t_n1_5` for the GR00T N1.5 adapter. Monitor and stop jobs
with `squeue -u "$USER"` and `scancel JOB_ID`. The writer creates
`episodes.csv/jsonl` and `summary.csv/json` incrementally; a complete 4-delay x
11-replan x 3-episode run has 132 episode rows and 44 conditions.

Plot a completed Xiaomi/GR00T task pair:

```bash
.conda-env/bin/python scripts/plot_stage1_delay_replan_heatmap.py \
  --run Xiaomi eval_results/stage1/xiaomi/PickPlaceCounterToSink \
  --run GR00T eval_results/stage1/gr00t_n1_5/PickPlaceCounterToSink \
  --task PickPlaceCounterToSink --episodes-per-cell 3 \
  --output eval_results/stage1/PickPlaceCounterToSink_xiaomi_gr00t.png
```

To inspect selected episodes as MP4, re-render a finished run:

```bash
.conda-env/bin/python scripts/replay_run_video.py \
  --run eval_results/stage1/xiaomi/PickPlaceCounterToSink \
  --out eval_results/stage1/videos/xiaomi --limit 2
```

### Gym wrapper
You can create environments using gym wrappers and run rollouts:
```py
import gymnasium as gym
import robocasa
from robocasa.utils.env_utils import run_random_rollouts

env = gym.make(
    "robocasa/PickPlaceCounterToCabinet",
    split="pretrain", # use 'pretrain' or 'target' kitchen scenes and objects
    seed=0 # seed environment as needed. set seed=None to run unseeded
)

# run rollouts with random actions and save video
run_random_rollouts(
    env, num_rollouts=3, num_steps=100, video_path="/tmp/test.mp4"
)
```

### Play back sample demonstrations of tasks
**(Mac users: for these scripts, prepend the "python" command with "mj": `mjpython ...`)**

Select a task and play back a sample demonstration for the selected task:
```
python -m robocasa.demos.demo_tasks
```

### Explore kitchen scenes
Explore 2500+ kitchen scenes:
```
python -m robocasa.demos.demo_kitchen_scenes
```

### Explore library of 2500+ objects
View and interact with both human-designed and AI-generated objects:
```
python -m robocasa.demos.demo_objects
```
Note: By default, this demo shows objaverse objects. To view AI-generated objects, add the flag `--obj_types aigen`.

### Teleoperate the robot
Control the robot directly, either through a keyboard controller or spacemouse. This script renders the robot semi-translucent in order to minimize occlusions and enable better visibility.
```
python -m robocasa.demos.demo_teleop
```
Note: If using SpaceMouse, you may need to modify the product ID to your appropriate model, setting `SPACEMOUSE_PRODUCT_ID` in `robocasa/macros_private.py`.

-------
## Tasks, datasets, policy learning, and additional use cases
Please refer to the [documentation page](https://robocasa.ai/docs/introduction/overview.html) for information about tasks, datasets, benchmarking, and more.

-------
## License
Code: [MIT License](https://opensource.org/license/mit)

Assets and Datasets: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)

-------
## Citation

**RoboCasa365:**

```bibtex
@inproceedings{robocasa365,
  title={RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots},
  author={Soroush Nasiriany and Sepehr Nasiriany and Abhiram Maddukuri and Yuke Zhu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

**RoboCasa (Original Release):**

```bibtex
@inproceedings{robocasa2024,
  title={RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots},
  author={Soroush Nasiriany and Abhiram Maddukuri and Lance Zhang and Adeet Parikh and Aaron Lo and Abhishek Joshi and Ajay Mandlekar and Yuke Zhu},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2024}
}
