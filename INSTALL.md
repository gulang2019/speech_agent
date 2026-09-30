# RoboCasa × 多模型延迟基准本地运行指南

本仓库在 RoboCasa 源码之上加入了 Xiaomi-Robotics-1、GR00T N1.5、pi0.5
和 Diffusion Policy 的本地单卡评测与 inference-delay 实验。文档分六部分：

1. 本地环境与资产准备。
2. RoboCasa 基础使用（建环境、跑 rollout、看任务）。
3. Xiaomi 模型加载、单条件评估、多参数网格实验。
4. 结果格式与统计分析方法。
5. 官方评估代码的差异。
6. 跨模型、跨任务延迟基准。

所有命令默认工作目录为仓库根目录：

```bash
cd /fact_home/xunyuanliu/dev/robo
```

## 1. 环境

本地已就绪的路径：

```text
.conda-env/                    # 已安装好 robocasa / robosuite / torch / transformers
.deps/robosuite/               # robosuite master 分支
models/xiaomi-robotics-1-robocasa/   # Xiaomi-Robotics-1-RoboCasa 权重快照
.deps/Xiaomi-Robotics-1/       # Xiaomi 官方源码，含 eval_robocasa 参考实现
.deps/Isaac-GR00T/              # GR00T N1.5 源码
.deps/openpi/                   # OpenPI 源码
.deps/diffusion_policy/         # Diffusion Policy 源码 checkout
.runtime/envs/groot/            # GR00T 隔离运行环境
.runtime/envs/openpi/           # pi0.5 隔离运行环境
.runtime/envs/diffusion_policy/ # Diffusion Policy 隔离运行环境
models/robocasa365_checkpoints/ # GR00T、pi0.5、Diffusion Policy checkpoint
models/clip-vit-large-patch14/  # Diffusion Policy 的本地 CLIP-L/14
```

三套模型依赖存在版本冲突，因此不会安装进 `.conda-env`。统一的 RoboCasa
主循环通过 `scripts/model_worker.py` 使用长度前缀 pickle 协议与隔离 worker
通信；worker stdout 保留给协议，模型日志写入 Slurm 的 `.err` 文件。
模型目录、隔离环境和日志都已加入 `.gitignore`。

激活环境，或直接用绝对路径解释器（推荐，脚本里也是这么做的）：

```bash
conda activate /fact_home/xunyuanliu/dev/robo/.conda-env
.conda-env/bin/python -c 'import robocasa, torch; print(robocasa.__file__, torch.__version__)'
```

### 厨房资产

资产已解压到 `robocasa/models/assets/`（约 23GB）。重建时在网络可用节点执行：

```bash
.conda-env/bin/python -m robocasa.scripts.setup_macros
.conda-env/bin/python -m robocasa.scripts.download_kitchen_assets
```

`DATASET_BASE_PATH` 只影响数据集目录，不影响厨房资产目录。

### 提交前自检

```bash
sbatch slurm/robocasa_smoke.sbatch      # GPU + MuJoCo EGL + 真实任务 reset/step
sbatch slurm/xiaomi_model_smoke.sbatch  # 模型加载 + 一次前向（合成图像，非成功率评测）
```

输出在 `logs/<job-name>-<jobid>.out|.err`。两个脚本默认 `dev` 分区、
`--gres=gpu:1`；换分区或指定卡型时改脚本里的 `#SBATCH` 行即可。

## 2. RoboCasa 基础使用

### 构造环境并跑随机 rollout

```python
import gymnasium as gym
import robocasa
from robocasa.utils.env_utils import run_random_rollouts

env = gym.make(
    "robocasa/PickPlaceCounterToCabinet",
    split="pretrain",   # pretrain / target
    seed=0,
)
run_random_rollouts(env, num_rollouts=3, num_steps=100, video_path="/tmp/test.mp4")
```

注意 `import robocasa` 必须早于 `gym.make`，否则任务不会注册。

### 常用调试入口

```bash
.conda-env/bin/python -m robocasa.demos.demo_tasks           # 回放任务示教
.conda-env/bin/python -m robocasa.demos.demo_kitchen_scenes  # 浏览厨房场景
.conda-env/bin/python -m robocasa.demos.demo_objects         # 浏览物体库
```

### 切换任务

实验脚本通过 `--task-name` 指定任务，可重复传入以串行评测多个任务。任务
地平线默认由 `robocasa.utils.dataset_registry_utils.get_task_horizon` 读取，
也可用 `--horizon` 覆盖。任务名必须是已注册的原子或复合任务，例如
`PickPlaceCounterToCabinet`、`OpenDrawer`、`TurnOnStove`。直接从注册表列
全部任务名（比翻渲染文档快）：

```bash
.conda-env/bin/python -c "
from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS, COMPOSITE_TASK_DATASETS
print('\n'.join(list(ATOMIC_TASK_DATASETS) + list(COMPOSITE_TASK_DATASETS)))" | sort
```

人类可读的任务清单在 `docs/tasks/atomic_tasks.md` 与
`docs/tasks/_generated/composite_tasks_details.md`。

RoboCasa 场景分布有两种：`--scene-set legacy5`（本地 Xiaomi 评测用的 5 个
固定 layout/style，配合 `--object-split pretrain|target`）或
`--scene-set pretrain|target`（整库场景分布）。地平线默认取注册表值
（例如 `PickPlaceCounterToCabinet` 是 750 步），而官方 `eval_robocasa`
对每个任务硬编码 300-1000 步的上限，两者不完全相同。

## 3. Xiaomi-Robotics-1 评测

实验入口是 `scripts/xiaomi_latency_experiment.py`。它在**单个 GPU 进程**里加载
模型，把推理放到后台线程，仿真主循环继续消费已有动作，从而模拟真实部署中的
异步推理延迟。模型动作块长度固定为 10。

### 参数语义

- `--delays-ms`：**注入**延迟，逗号分隔。语义由 `--delay-domain` 决定。
- `--delay-domain`：`sim`（默认）或 `wall`。
  - `sim`：延迟按控制步解析，`delay_steps = round(delay_ms * control_frequency / 1000)`。
    响应严格在请求后 `delay_steps` 步才可安装；模型真实耗时只影响墙钟，
    不影响 stale 步数，因此注入延迟是唯一自变量，四个模型可直接对齐比较。
    模型比延迟慢时主循环阻塞等待，等待时长记入 `mean_block_wait_ms`。
  - `wall`：旧语义，worker 在真实模型耗时之上再 `sleep(delay_ms)`，
    实际 stale 依赖「模型耗时 + 注入延迟」。用于复现归档结果。
- `--replan-steps-list`：每隔多少个控制步发起一次新推理；有效范围 `1..10`
  （等于动作块长度）。
- `--rtc`：`off,on`。`off` 表示响应到达后从新动作块第 0 步直接替换排队动作；
  `on` 表示按响应到达时刻对齐、丢弃过期动作，并用 `--rtc-blend-steps` 做线性融合。
- `--episodes`：每个条件（delay × replan × rtc 的笛卡尔积）的 episode 数。
- `--seed`：起始 episode seed，所有条件复用同一组 seed，便于逐 episode 配对比较。

### 先做参数检查

`--dry-run` 不加载模型，只打印将要执行的条件组合：

```bash
.conda-env/bin/python scripts/xiaomi_latency_experiment.py \
  --task-name PickPlaceCounterToCabinet \
  --delays-ms 0,100 --replan-steps-list 5,10 --rtc off,on \
  --episodes 5 --dry-run
```

### 单条件基线评估

```bash
.conda-env/bin/python scripts/xiaomi_latency_experiment.py \
  --model-path models/xiaomi-robotics-1-robocasa \
  --task-name PickPlaceCounterToCabinet \
  --scene-set legacy5 --object-split pretrain \
  --episodes 50 \
  --delays-ms 0 --replan-steps-list 5 --rtc off \
  --output-dir eval_results/baseline_50ep
```

参数较多时推荐走 Slurm 包装脚本，环境变量即可覆盖：

```bash
sbatch slurm/xiaomi_latency_experiment.sbatch
```

支持的覆盖变量：`MODEL_NAME`、`MODEL_PATH`、`MODEL_ENV`、`TASK_NAME`、
`TASK_SET`、`SCENE_SET`、`OBJECT_SPLIT`、`EPISODES`、`DELAY_DOMAIN`、
`DELAYS_MS`、`REPLAN_STEPS_LIST`、`RTC`、`HORIZON`、`CONTROL_FREQUENCY`、
`OUTPUT_DIR`、`DIFFUSION_STEPS`、`ATTN_IMPLEMENTATION`。

### 多参数网格实验

脚本对 `delays × replans × rtc` 做全笛卡尔积，所以“固定两个变量、只变一个”
就是把另外两个变量传成单值。三个单变量实验如下（每个条件 50 episodes，
约 30-50s/episode，单卡总耗时 1-5 小时）：

```bash
# 实验 A：delay 0-600ms，固定 replan=5、rtc=off
DELAYS_MS=0,100,200,300,400,500,600 REPLAN_STEPS_LIST=5 RTC=off \
EPISODES=50 OUTPUT_DIR=eval_results/xiaomi_ofat_delay_r5_rtcoff_50ep \
  sbatch slurm/xiaomi_latency_experiment.sbatch

# 实验 B：replan 1-10，固定 delay=0、rtc=off
DELAYS_MS=0 REPLAN_STEPS_LIST=1,2,3,4,5,6,7,8,9,10 RTC=off \
EPISODES=50 OUTPUT_DIR=eval_results/xiaomi_ofat_replan_d0_rtcoff_50ep \
  sbatch slurm/xiaomi_latency_experiment.sbatch

# 实验 C：rtc on/off，固定 delay=0、replan=5
DELAYS_MS=0 REPLAN_STEPS_LIST=5 RTC=off,on \
EPISODES=50 OUTPUT_DIR=eval_results/xiaomi_ofat_rtc_d0_r5_50ep \
  sbatch slurm/xiaomi_latency_experiment.sbatch
```

三个实验可以同时提交到不同 GPU；脚本每写一个 episode 就刷新一次输出文件，
中途失败可从 `episodes.csv` 看到已完成进度。

零交互的二维/三维全网格也直接支持，例如 delay × replan 全扫描：

```bash
DELAYS_MS=0,100,200,300,400,500,600 \
REPLAN_STEPS_LIST=1,2,5,10 RTC=off EPISODES=50 \
OUTPUT_DIR=eval_results/xiaomi_grid_delay_replan \
  sbatch slurm/xiaomi_latency_experiment.sbatch
```

若要同时评测多个任务，直接调用脚本并重复传 `--task-name`（sbatch 包装只
接受单个 `TASK_NAME`）；此时条件数再乘以任务数。

## 4. 结果与统计

每次运行输出到 `--output-dir`：

```text
episodes.jsonl / episodes.csv   # 每条 episode 一行，含 seed、layout/style id、成功、步数、延迟统计
summary.json / summary.csv      # 按条件聚合：成功率、95% Wilson 区间、平均步数、过期/替换动作数
```

延迟相关字段（`sim` 域）：

- `delay_domain` / `delay_steps`：条件实际使用的延迟域与等效控制步数（`--delay-domain sim`）。
- `mean_arrival_age_steps` / `max_arrival_age_steps`：响应到达时相对请求时刻的
  步龄；`sim` 域下应稳定等于 `delay_steps`，是「延迟确实生效」的直接证据。
- `stale_actions_discarded`：RTC 打开时被丢弃的过期动作数，`sim` 域下由
  `delay_steps` 决定，可精确复现。
- `mean_block_wait_ms`：模型比注入延迟更慢时主循环的额外阻塞时长；模型真实
  耗时本身记在 `mean_model_latency_ms`，二者都不参与 stale 计算。

`wall` 域只额外保留 `mean_total_inference_latency_ms` 的实际含义
（模型耗时 + 注入延迟），以便与 `eval_results/archive/legacy_wallclock_delay/`
中的旧结果对照。

`summary.csv` 的成功率区间是二项 Wilson 区间，适合单条件描述。要比较条件之间
的差异，用 `scripts/analyze_xiaomi_ofat.py`：它要求三组 OFAT 目录分别对应
delay / replan / rtc 扫描，逐 episode 按 `(task, episode, seed, layout, style)`
配对，输出配对 bootstrap 差异区间、精确 McNemar 检验和 Holm 校正：

```bash
.conda-env/bin/python scripts/analyze_xiaomi_ofat.py \
  --delay-dir  eval_results/xiaomi_ofat_delay_r5_rtcoff_50ep \
  --replan-dir eval_results/xiaomi_ofat_replan_d0_rtcoff_50ep \
  --rtc-dir    eval_results/xiaomi_ofat_rtc_d0_r5_50ep \
  --episodes 50 \
  --output-dir eval_results/xiaomi_ofat_analysis_50ep
```

产出 `ofat_analysis.csv` 和 `report.md`。脚本会校验固定变量、条件取值集合、
每个条件的 episode 数以及是否存在 error，任一不满足即报错，避免误分析。

### 提高统计可信度

- 每个条件至少 50 个 episode；比较多个条件时注意 Holm 校正会随条件数变严。
- 所有条件共用同一组 seed 才能配对比较，不要给不同条件传不同 `--seed`。
- 关注 `errors` 是否为 0、`mean_fallback_steps` 是否异常高，以及 `sim` 域下
  `mean_arrival_age_steps` 是否等于条件声明步数，确认延迟真的生效。
- 50% 左右成功率附近的 Wilson 区间约 ±14 个百分点，小差异需要更多 episode
  或跨任务重复才能下结论。

## 5. 与官方评估代码的关系

`.deps/Xiaomi-Robotics-1/eval_robocasa/README.md` 是官方 client/server 多卡
实现（8 server + 8 client，24 任务 × 100 episodes，官方平均 74.2%）。
本仓库的脚本是独立单卡实现，便于快速做 delay / replan / RTC 这类受控变量实验。
与官方一致的部分：三路相机名与 256 分辨率、crop ratio 0.95、动作块 10、
`robocasa_mg` 机器人类型，以及 `legacy5` 对应的 5 组 layout/style。

与官方不同的部分：

- 官方用固定的 300-1000 步任务上限，本地默认取注册表 horizon（750 步）；
- 官方的 `obj_instance_split="B"` 在本仓库注册表里对应
  `--object-split pretrain`；
- 官方是多卡 client/server，本地是单卡同进程异步推理，seed 也不同。

因此绝对成功率和官方 74.2% 不可直接对齐，跨实现比较时应固定 `--seed`、
`--scene-set` 和 `--horizon`，只比较条件之间的相对差异。

## 6. 跨模型、跨任务延迟基准

当前已准备好并通过 synthetic forward smoke 与真实 RoboCasa 单回合接口 smoke 的模型：

- `xiaomi`：本地 Xiaomi-Robotics-1 checkpoint，10 步 action chunk；
- `gr00t_n1_5`：RoboCasa 官方 GR00T N1.5 checkpoint，16 步 native chunk；
- `pi0_5`：RoboCasa 官方 pi0.5 checkpoint，50 步 native chunk；
- `diffusion_policy`：RoboCasa 官方 hybrid Transformer checkpoint，8 步 native chunk。

`pi0` 和 target fine-tuned 检查点仍列为后续扩展。模型注册表位于
`scripts/latency_model_adapters.py`；未下载的模型保持 `planned`，不会静默回退到
其他模型。

先做三套合成输入 smoke（每个模型需要一张 GPU）：

```bash
MODEL_NAME=gr00t_n1_5 sbatch slurm/model_adapter_smoke.sbatch
MODEL_NAME=pi0_5 sbatch slurm/model_adapter_smoke.sbatch
MODEL_NAME=diffusion_policy sbatch slurm/model_adapter_smoke.sbatch
```

输出应分别包含 `[16, 12]`、`[50, 12]`、`[8, 12]`，并且 `finite=true`
（`xiaomi` 为 `[10, 7]`）。再做真实
仿真接口 smoke；`HORIZON=20` 只验证 reset/render/infer/step 链路，不用于评估
成功率：

```bash
MODEL_NAME=gr00t_n1_5 TASK_NAME=PickPlaceCounterToCabinet \
SCENE_SET=legacy5 DELAYS_MS=0 REPLAN_STEPS_LIST=5 RTC=off EPISODES=1 \
HORIZON=20 OUTPUT_DIR=eval_results/gr00t_interface_smoke \
  sbatch slurm/xiaomi_latency_experiment.sbatch

MODEL_NAME=pi0_5 TASK_NAME=PickPlaceCounterToCabinet \
SCENE_SET=legacy5 DELAYS_MS=0 REPLAN_STEPS_LIST=10 RTC=off EPISODES=1 \
HORIZON=20 OUTPUT_DIR=eval_results/pi05_interface_smoke \
  sbatch slurm/xiaomi_latency_experiment.sbatch

MODEL_NAME=diffusion_policy TASK_NAME=PickPlaceCounterToCabinet \
SCENE_SET=legacy5 DELAYS_MS=0 REPLAN_STEPS_LIST=8 RTC=off EPISODES=1 \
HORIZON=20 OUTPUT_DIR=eval_results/dp_interface_smoke \
  sbatch slurm/xiaomi_latency_experiment.sbatch
```

任务、场景和延迟档位统一记录在
`configs/latency_benchmark_v1.json`：

```bash
.conda-env/bin/python scripts/validate_latency_manifest.py
```

其中包含：

- `atomic_screening`：8 个原子任务，用于第一轮筛选；
- `atomic_final`：6 个原子任务，用于完整 50-episode 实验；
- `composite_extension`：4 个复合任务，用于第二阶段；
- `legacy5`：与当前 Xiaomi 结果兼容的 5 个固定场景；
- `pretrain20`：用于跨模型主结果的 20 个固定预训练场景；
- `target10`：10 个 target 厨房场景，用于独立泛化实验。

Xiaomi dry-run 示例：

```bash
.conda-env/bin/python scripts/xiaomi_latency_experiment.py \
  --model-name xiaomi \
  --task-set atomic_screening \
  --scene-set pretrain20 \
  --delays-ms 0,100,300,500 \
  --replan-steps-list 5 \
  --rtc off \
  --episodes 5 \
  --dry-run
```

Slurm 中可以使用同一套配置：

```bash
MODEL_NAME=xiaomi \
TASK_SET=atomic_screening \
SCENE_SET=pretrain20 \
DELAYS_MS=0,100,300,500 \
REPLAN_STEPS_LIST=5 RTC=off EPISODES=20 \
OUTPUT_DIR=eval_results/latency_screening_xiaomi \
  sbatch slurm/xiaomi_latency_experiment.sbatch
```

不同模型的主比较应同时记录额外 delay、真实模型耗时、总响应延迟、fallback
steps、stale actions 和 episode success。由于各模型 native action chunk 与单次
推理耗时不同，跨模型主表应同时报告 native chunk 以及 replan interval；如需公平
比较“总响应延迟”，使用 manifest 中的 `normalized_total_latency_targets_ms`
设计额外 delay 条件。

### 动作空间保真（重要）

`PandaOmron` 的 dense action 是 **12 维**：

```text
[0:6]   end_effector_position(3) + end_effector_rotation(3)
[6:7]   gripper_close
[7:11]  base_motion(4)
[11:12] control_mode
```

三个 RoboCasa365 官方 checkpoint（`gr00t_n1_5`、`pi0_5`、`diffusion_policy`）
的 native 输出同样是 12 维，顺序与官方 `robocasa.utils.env_utils.convert_action`
一致；早期实现把它截断成 7 维（只留手臂与夹爪），会丢掉基座运动与
`control_mode`，并使两个二值通道落到 `0`——而夹爪控制器用 `np.sign(action)`，
`0` 等于夹爪完全不动。

现在的约定：适配器原样返回 native 宽度，`scripts/xiaomi_latency_experiment.py`
的 `make_action` 负责落到 dense action，并对 `gripper_close` / `control_mode`
按官方 `PandaOmronKeyConverter.unmap_action` 的阈值语义做 `<0.5 → -1`、
否则 `+1` 的映射。改动适配器动作宽度时必须同步核对这两处，
`tests/test_xiaomi_latency_experiment.py::test_make_action_*` 覆盖了该契约。

## 7. 跨模型主网格（post-fix）

修复动作空间保真问题后，主比较由 `slurm/submit_sim_delay_grid.sh` 提交：每个
`(model, task)` 一个作业，结果落在 `eval_results/simdelay_grid/<model>/<task>/`，
失败只影响一个切片，且可并行铺满集群。

```bash
bash slurm/submit_sim_delay_grid.sh        # 4 模型 x 8 任务，30 ep/条件
MODELS="pi0_5" TASKS="OpenDrawer" EPISODES=5 bash slurm/submit_sim_delay_grid.sh
```

默认 `SCENE_SET=pretrain20`（20 组固定 layout/style）、`DELAYS_MS=0,100,300,500`、
`REPLAN_STEPS_LIST=5`、`RTC=off`。已完成切片会被自动跳过，可重复执行补跑。

汇总用：

```bash
.conda-env/bin/python scripts/analyze_sim_delay_grid.py \
  --root eval_results/simdelay_grid \
  --output-dir eval_results/simdelay_grid_analysis
```

产出 `cross_model_grid.csv`（每个 model/task/delay 一行，配对 bootstrap 差异区间
+ 精确 McNemar，Holm 校正）、`macro_by_model.csv`（按任务宏平均，避免简单任务
主导）和 `SUMMARY.md`。脚本会校验 `replan/rtc` 固定、各延迟档 episode 数一致、
无 error，并在 `sim` 域下断言 `mean_arrival_age_steps == delay_steps`。

注意 `diffusion_policy` 单次推理约 480 ms，一个 120-episode 切片约 3.4 小时，
是最慢的一档；其余三模型在 1-1.5 小时内。

### 场景固定（重要）

`Kitchen._setup_model` 每次执行都会 `self.rng.choice(self.layout_and_style_ids)`
重新采样 layout/style，而 `_load_model` 在夹具放置失败时会重建模型并再次执行它。
重试次数取决于解释器状态，因此**同一个 seed 在不同运行里可能落到不同厨房**——
同一条件内还算可复现，跨条件配对时会退化成「不同场景比成功率」。

`pin_scene()` 现在按 `layout_and_style_ids[(seed + episode) % n]` 预先解析出
`layout_ids`/`style_ids` 传给 `create_env`，使场景成为 (seed, episode) 的纯函数。
30 个 episode 会均匀覆盖 manifest 里的全部场景。

用 `scripts/check_scene_pairing.py` 检查一个已完成网格是否存在跨延迟场景漂移：

```bash
.conda-env/bin/python scripts/check_scene_pairing.py --root eval_results/simdelay_grid
```

修复前的网格有 13/960（1.35%）的 episode 组出现漂移，集中在
`PickPlaceCounterToSink`；该任务四个模型已用修复后的代码重跑，旧结果保留在
`eval_results/simdelay_grid_unpinned/` 以便对照。
