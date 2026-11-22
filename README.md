> 基于 [Zhengsw03/RL-Token-Pi05-open](https://github.com/Zhengsw03/RL-Token-Pi05-open) 的 SO-101 复现，增加 ARX X5 控制接口、动作转换和人工接管。在线策略更新的独立评估尚未完成。

ARX 训练与上机命令见下文。各轮实验改了什么、有哪些实测依据，见 [Stage 2 算法与控制迭代](#stage-2-算法与控制迭代)。

## ARX X5 配置与运行

π0.5 数据与微调权重由 [pi0.5](https://github.com/lanxinmob/pi0.5) 准备。SDK 环境路径填写在 `config.json`，VLA、编码器、分类器、相机及硬件参数填写在 `rlt_stage2.json`。

```bash
bash setup_arx.sh
./rlt check --gpu
./rlt stage2-check --auto-critical
./rlt stage2 --auto-critical --episodes 5
```

最后一条会连接配置中的执行臂和相机。启动前核对 CAN、标定与工作空间，停止占用同一机械臂的其他控制程序。Stage 2 使用 `.venv`，硬件进程使用配置中指定的 SDK Python。

`rlt_stage2.json` 中的 `actor_handoff_steps` 默认为 5。VLA → Actor、Actor → Actor 和 Actor → VLA
换块时，六个关节从控制器最后接受的位置出发，用最近两条已接受指令估算起始速度，做三次插值。
第 5 个控制周期接回新块原定的位置和速度（30 Hz 下约 167 ms），夹爪沿用原定时序。
设为 0 可关闭换块过渡。

每回合起步或结束人工接管（`h` 或 `s`）时，等待新计划期间的 `HOLD` 保持切回策略时的关节和夹爪位置。
`start_blend_steps` 控制新计划的起步过渡。ARX 默认 6 条指令，第 1 条关节指令保持该姿态，随后平滑接近计划轨迹，第 6 条接回计划目标。夹爪按新计划的时序执行。
命令行 `--start-blend-steps N` 可覆盖默认值，0 或 1 关闭起步过渡。

ARX 在线训练每次启动或续训都由 VLA 执行。进入关键阶段后可用 `h` 人工纠正，按 `f` 后用
`y` 或 `n` 保留回合和反馈。满足 warmup 和回放条件后，Actor 与 Critic 在后台更新。
执行 VLA 的关键阶段，learner 用同一观测评估 Actor，在本次输出目录写入 `actor_shadow.jsonl`；
可用 `tail -f "$RUN/actor_shadow.jsonl"` 跟踪，其中 `RUN` 是本次运行的输出目录。

每行对应一个 VLA 窗口。`actor` 按当前执行采样配置生成候选，`mean` 使用 Actor 均值，
`vla` 是对照。三者的 `peak_step_deg` 和 `over_limit_steps` 覆盖整块六个关节的限速前指令，
包含从上一条已接受指令到首帧的变化。`peak_actor_vla_deg` 记录同一控制步最大的关节目标偏差；
夹爪偏差单独以毫米计。观察不同起始姿态与抓取、放置阶段的这些数值后，由操作者按 `a`
允许 Actor 接管。warmup 结束且达到所需更新次数后，Actor 从下一个符合条件的关键阶段动作块开始执行。
再按 `a` 可在后续动作块交回 VLA，按 `h` 可随时请求人工接管。影子评估的候选只用于比较，回放记录实机实际执行的数据。

每回合结束、评分前，终端、`train.log` 和 `rl_metrics.log` 会输出一行 `[SHADOW]`。
对 `mean`、`actor` 和 `vla` 分别取本回合最大的 `peak_step_deg`，并汇总
`over_limit_steps` 为「超限步数/已评估控制步数」。同一步多个关节超限只计一次。
参考范围取自本次控制器的关节限速。默认 1 rad/s、30 Hz 时，单关节每步为 0 至约 1.91°，
超限步数目标为 0。`valid_windows` 显示有效窗口数/提交窗口数。无有效窗口时显示无数据。

要在 Rerun 中查看相机、VLA 目标、Actor 选中目标和 SDK 实际接受的目标，可运行：

```bash
./rlt stage2 --auto-critical --episodes 200 --display-data \
  --display-fps 10 --display-compressed-images
```

`./rlt stage2` 退出时在终端打印 `[RESUME]`、下次回合编号和一条可直接执行的续训命令。
命令沿用本次训练参数，下一次运行会创建新的输出目录。正常结束使用最终 checkpoint；
训练子进程异常退出时，启动器依次检查本次的最终 checkpoint、最近的定期 checkpoint 和启动时的恢复点。
若已达到总回合目标，打印的命令默认再运行一回合，可调整 `--episodes` 继续到其他总回合数。
每次启用 `--display-data` 都会在本次输出目录生成 `stage2_visualization.rrd`，即使 Viewer
启动失败也会保留该文件。训练、推理和评估使用相同的结构化视图，完成后可直接打开：

```bash
.venv/bin/rerun outputs/rlt/YOUR_RUN/stage2_visualization.rrd
```

Viewer 中的 `Live camera` 显示腕部相机，`Joint targets (deg)` 同时显示 VLA、Actor
选中值和 SDK 接受值，`Measured vs accepted (deg)` 显示实际关节状态与发送目标，
`Joint target error (deg)` 显示 `selected-vla` 和 `accepted-selected`，其余视图显示夹爪、
控制来源、关键阶段、更新次数、Q 值、Critic loss、Actor loss、TD error、BC loss、Actor advantage 和 shadow 指标。
ARX 记录将弧度/米转换为度/毫米，SO-101 记录保留度数。相机、目标和指标使用同一条
`control` 时间线，拖动时间轴即可逐步对齐动作与画面。

推理脚本默认也使用结构化记录，需指定独立输出目录，例如
`scripts/infer_rlt_stage2_pi05.py --display_data --output_dir outputs/rlt/deploy_run`。
评估脚本使用 `--display_data --output_dir outputs/rlt/eval_run`。使用 `--display-fps`
调整刷新率，使用 `--display-compressed-images` 节省带宽。可用
`../pi05/.tools/bin/uv pip install --python .venv/bin/python 'rerun-sdk>=0.24,<0.34'` 安装 Rerun SDK。
SDK 安装后还需要同版本的 Rerun Viewer。Linux x86_64 可下载
`rerun-cli-0.33.1-x86_64-unknown-linux-gnu` 放到 `~/.local/bin/rerun-cli` 并加执行权限，
或者使用 `cargo binstall --force rerun-cli@0.33.1`。

使用已有 replay 离线训练零初始化的 residual Actor 和新的 Twin Critic：

```bash
./rlt stage2-retrain \
  --source-checkpoint outputs/rlt/YOUR_RUN/final_checkpoint.pt \
  --actor-residual-scale 0.05 \
  --output outputs/rlt/stage2_retrain_YYYYMMDD_HHMMSS
```

Actor 的网络均值为 `VLA参考 + scale * tanh(残差网络输出)`。最后一层的权重和偏置从零开始。
默认 `scale=0.05`，单位是每个动作维度的归一化值。50% reference dropout 只屏蔽残差网络看到的参考，
加到输出上的 VLA 参考始终保留。新回放分别记录 VLA 输入和人工纠正目标，旧回放沿用当时保存的参考。
默认每条回放对应 5 次 Critic 更新，每两次 Critic 更新一次 Actor，使用
`-Q + beta * MSE(actor_action, replay_reference)`（默认 `beta=10`）。
重训将 checkpoint 和 replay journal 写入新目录。用 `--actor-residual-scale 0` 可训练原全量输出 Actor。

离线日志按 Critic 更新次数记录训练 batch 的指标。用下面的命令生成交互曲线，包括实际 Actor loss、双 Critic 的 TD 均方误差之和、行为克隆 MSE，以及同一 batch 上 Actor 均值动作与回放参考动作的 Q 值差。残差策略用原始 VLA 动作作参考，缺少原始 VLA 的旧回放使用当时保存的参考动作，其中可能含人工纠正。

```bash
RUN=outputs/rlt/YOUR_RETRAIN_RUN
.venv/bin/python scripts/plot_rl_metrics.py "$RUN/offline_metrics.jsonl" --out "$RUN/rl_losses_history.html"
```

冻结权重在 ARX 上评估时，`--episodes` 沿用总回合编号。下面从第 200 回合 checkpoint 开始测 10 回合，沿用采集时的起步过渡和 RTC 参数。自动 critical 开启后，Actor 默认获准在关键阶段执行。每回合按 `f` 结束、用 `y/n` 评分，评估目录的 `training_log.json` 在 `inference_scores` 中单独保存这次的结果。

```bash
./rlt stage2 --eval-only --auto-critical --actor-execute-mean --episodes 210 \
  --resume-from "$RUN/final_checkpoint.pt" --start-blend-steps 3 --rtc-max-guidance-weight 8
```

单臂 X5_umi 的默认夹爪速度采用 SDK 原生值 0.30 m/s，30 Hz 下每条夹爪指令最多移动 10 mm。
从旧 checkpoint 对照试验时，`--gripper-speed` 指定离线训练模拟的有效夹爪速度（m/s），
省略时沿用采集时的值。历史 replay 中的动作和奖励保留采集时的记录。上机时执行速度取自
`rlt_stage2.json` 的 `hardware_config.gripper_speed` 并受 SDK 原生值约束。

下面从源 checkpoint 的 Actor、Critic 及各自的 target 权重开始微调，将 residual 上限放宽到 0.10。
优化器与本次更新计数从零开始，回合编号和 replay 接续源 checkpoint。

```bash
./rlt stage2-retrain --source-checkpoint outputs/rlt/YOUR_RUN/final_checkpoint.pt \
  --init-from-source --gripper-speed 0.30 --actor-residual-scale 0.10 \
  --updates 1200 --output outputs/rlt/stage2_gripper_residual
```

在线训练的 replay batch 固定由 30% warmup/offline 数据和 70% online/HIL 数据组成。
warmup/offline 池均匀采样。online/HIL 池按新旧 transition 的插入年龄衰减，并结合每条数据已经被采样的次数做平衡，
同时保留 20% 的均匀概率下限，旧数据不会被永久饿死。采样计数随 checkpoint 保存并在续训时恢复。
默认衰减常数为 2000 条 transition，参数为 `replay_online_decay_steps`。

从训练结果继续在线采集时，Stage 2 从 checkpoint 读取 Actor 类型和残差上限。
要从下个回合开始收紧已有 residual Actor，可在续训命令中追加
`--resume-actor-residual-scale 0.02`。Actor 和 target Actor 都使用新上限，
优化器、更新计数和 replay 按原 checkpoint 恢复，新保存的 checkpoint 会记录该上限。
需要更大的纠偏空间时，可以同时传 `--resume-actor-residual-scale 0.12 --actor-residual-step 0.006`。
第一项限制每个通道相对 VLA 的总偏移，第二项限制相邻已发送控制步的偏移变化，两者使用归一化动作单位。
每个短计划取平滑、插值后的平均修正为目标，逐步靠近该目标。跨计划沿用上一条已发送指令的修正；
关闭 Actor 后逐步回到零，人工接管直接交给操作者，重新进入策略时从零开始。VLA 自身的运动仍受 SDK 执行约束。
30 Hz 下从零达到 `0.12` 最快需要 20 条指令，约 0.67 秒。旧 replay 未记录上一条修正时，
训练在对应采样窗口用零作起点。开启此模式的 checkpoint 使用 `./rlt stage2` 的控制时间线执行。
离线训练参数为 `--actor-residual-scale 0.12 --actor-residual-step 0.006`。
续训时省略步长参数会沿用 checkpoint；旧 checkpoint 默认步长限制关闭。
用 `--actor-execute-mean` 可直接评估均值，去除固定标准差的探索采样。下面的起步步数 3 对应已有 beta10 回放。

```bash
./rlt stage2 --auto-critical --episodes 200 --actor-execute-mean \
  --resume-from outputs/rlt/stage2_retrain_YYYYMMDD_HHMMSS/final_checkpoint.pt \
  --start-blend-steps 3 --rtc-max-guidance-weight 8
```

更换 VLA 或示教数据后，重新提取特征并训练配套 Stage 1：

```bash
./rlt precompute --output outputs/rlt/embeddings_new
./rlt stage1 --embeddings outputs/rlt/embeddings_new \
  --output outputs/rlt/stage1_new --steps 30000
```

运行时 `c` 切换关键阶段，`f` 结束并评价回合，随后用 `y/n/d` 标记成功、失败或丢弃，终端 `q` 正常退出。启用 `--auto_critical` 后，分类器用 `threshold_on` 自动进入、用 `threshold_off` 自动退出，默认每回合只自动进入一次；手动关闭后不会在同一回合重新触发。人工接管结束后从新的实测状态重新规划。

## Stage 2 算法与控制迭代

ARX 抓取实验从 50 条示教、28,000 步的 π0.5 和 30,000 步的 Stage 1 开始。下面记录 Stage 2 中改动过的训练目标、真机控制和实验参数。

### Actor 与 Critic 如何使用真机数据

| 时间 | 训练中发现的问题 | 对算法的修改 |
| --- | --- | --- |
| 10 月底至 11 月 9 日 | 提前结束的动作块含无效尾帧。Actor 查询 Q 时还能利用这些帧。另一个偏差来自真机对请求动作做了滤波和限速。 | 用有效步掩码屏蔽尾帧。Critic 学习控制器实际接受的动作，Actor 损失和 TD 目标查询候选动作时也走同一套可微动作转换。 |
| 11 月 21 日 | Critical 段里既有策略动作，也有人工接管。需要明确每种动作对训练的作用。 | Critic 用实际动作和按有效步折扣的回报拟合 TD 目标。人工操作所在步用人工指令替换行为克隆目标。Actor 同时优化采样动作的 Q 和带掩码的行为克隆 MSE。离线实验先后使用 BC 权重 2.5 和 10，后续 200 回合实验采用 10。 |
| 11 月 22 日 | 全量输出的 Actor 可以一次大幅偏离 VLA。 | 改为有界残差 Actor，以 VLA 为起点，残差输出层零初始化。新 replay 分别保存原始 VLA 计划与人工纠正目标，让 Actor 在部署时也能从原始 VLA 计划出发。 |
| 11 月 22 日 | 在线续训需要兼顾早期经验和刚采集的数据。 | 在线 batch 给 warmup/offline 池分配 30%、online/HIL 池分配 70%，后者按年龄和已采样次数调整概率。固定回放离线重训使用均匀采样。 |

### 从影子评估到真机接管

- **先观察候选动作。** 11 月 20 日起，Rerun 把 VLA 参考、选中目标、SDK 接受目标和实测状态放在同一控制时间线上。随后增加关键阶段的 shadow 评估。第 45 回合的 310 个影子控制步里，Actor 候选的最大关节跳变为 8.30°，269 步超过当时的 1.91°/步参考值；同期 VLA 最大 0.66°，超限 0 步。这些数值测的是限速前的指令。
- **把接管时机交给操作者和关键阶段检测器。** 早期尝试用 Actor 均值动作相对 VLA 的 Q 优势决定能否接管，后来移除了这道门槛。在线训练时，操作者按 `a` 授权，训练就绪并进入 critical 后才执行 Actor。`c` 可手动切换 critical，自动模式使用进入和退出两个阈值，`h` 可请求 HIL。
- **分别处理起步、换块和人工交还。** 等待新计划时保持交还姿态。起步用 `start_blend_steps` 渐入（ARX 默认 6，本次实验用 3）；涉及 Actor 的策略换块用最近两条已接受指令估计起始速度，六个关节经过 5 个控制步的三次插值接回新轨迹。夹爪保留计划时序。
- **单独校准夹爪执行速度。** 第 103 回合后用 33 个回放窗口、963 个控制步比较了夹爪设置和残差幅度。X5_umi 夹爪从 0.01 m/s 调到 SDK 原生的 0.30 m/s，并将这个速度用于离线动作转换，便于比较请求与接受动作的差距。

### 残差幅度试验

| 回放截至 | 实际做过的改动 |
| --- | --- |
| 第 103 回合 | 在夹爪速度实验中比较残差上限 0.05 和 0.10，观察放宽上限后请求动作与控制器接受动作的差距。 |
| 第 131 回合 | 从已有权重分别以 0.05 和 0.015 微调 1,200 次 Critic 更新；另用 0.03 在截至该回合的全部 replay 上从头重训 Actor 和 Critic。 |
| 第 135 回合 | 将上限收紧到 0.02，从已有权重微调，只用第 132–135 回合的 691 个窗口，做 120 次 Critic 更新。 |
| 第 155 回合 | 总幅度放宽到 0.12，相邻控制步的残差变化限制为 0.006。每个计划取平均修正为目标，按控制步逐渐靠近；换块时继承上一条修正，持续同方向的修正可以累积。 |

第 155 回合的离线对照覆盖 27 个窗口、861 个控制步。前一列指标衡量 Actor 能把关节目标移离参考多远，后一列衡量修正量在相邻两步之间最多跳多少。

| 归一化幅度与逐步约束 | 关节修正幅度 P95 | 最大单步关节修正变化 |
| --- | ---: | ---: |
| 0.02，仅限制总幅度 | 1.07° | 0.69° |
| 0.12，仅限制总幅度 | 5.38° | 2.29° |
| 0.12，每步最多变化 0.006 | 3.19° | 0.32° |

### 第 200 回合的离线结果与评估

从第 1–200 回合的 36,712 个执行窗口重新初始化 Actor 和双 Critic。窗口来源包括 12,362 个 VLA、14,216 个 Actor、7,497 个全人工和 2,637 个人工与策略混合窗口。此次保留总幅度 0.12、逐步变化上限 0.006；batch 为 256，BC 权重为 10，共完成 183,560 次 Critic 和 91,780 次 Actor 更新。

交互报告保留四条曲线。最后 100 个已记录训练 batch 的均值为 Actor loss **−0.08026**、双 Critic TD loss **0.00172**、行为克隆 MSE **0.00301**、同批次 Q 差 **+0.02884**。Actor 均值动作的 Q 为 0.11088，回放参考动作为 0.08204。部分早期 HIL 窗口的参考包含人工纠正。上述数值描述 Critic 在训练回放上的估计；真机成功率需由冻结策略的新回合评分统计。

本次重训使用的运行目录为 `outputs/rlt/stage2_ep200_retrain_20261123`。将上文的 `RUN` 设为这个目录即可重绘 `rl_losses_history.html`，或用 `--eval-only --actor-execute-mean` 测第 201–210 回合。评估评分单独写入本次评估目录的 `training_log.json`。

下文保留 SO-101 原项目介绍；其中演示视频与结果对应原项目。

---

<h1 align="center">RL-Token-Pi05</h1>

<p align="center">
  <b>SO-101 机械臂上的 RL Token 全流程</b><br>
  <sub>采集示教 · 微调 π0.5 · 训练 RL Token · 关键阶段分类器 · 真机在线 RL · 推理部署</sub>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.12-blue" alt="Python 3.12">
  <a href="https://github.com/huggingface/lerobot"><img src="https://img.shields.io/badge/LeRobot-0.6.2-ffcc00" alt="LeRobot 0.6.2"></a>
  <img src="https://img.shields.io/badge/VLA-%CF%800.5-8a2be2" alt="pi0.5">
  <img src="https://img.shields.io/badge/robot-SO--101-lightgrey" alt="SO-101">
</p>

<p align="center"><a href="README.md">English</a> | <b>中文</b></p>

<details>
<summary>目录</summary>

- [流程](#流程)
- [安装](#安装)
- [步骤 0：硬件准备](#步骤-0硬件准备)
- [步骤 1：采集示教数据](#步骤-1采集示教数据)
- [步骤 2：微调 VLA](#步骤-2微调-vla)
- [步骤 3：训练 RL Token](#步骤-3训练-rl-token)
- [步骤 4：训练关键阶段分类器](#步骤-4训练关键阶段分类器)
- [步骤 5：真机在线 RL](#步骤-5真机在线-rl)
- [步骤 6：推理与部署](#步骤-6推理与部署)
- [配置项](#配置项)
- [出问题怎么办](#出问题怎么办)
- [致谢](#致谢)
- [贡献指南](#贡献指南)

</details>

<p align="center">
  <img src="assets/rl_token_effect_comparison.gif" alt="效果对比" width="780">
</p>

<p align="center"><i>视频把 VLA 基线与 RL Token 策略在同一任务上的表现并排对比，两块画面分别在自己的结论时刻定格：
左侧停在失败的尝试上，右侧停在成功的尝试上。</i></p>

RL-Token-Pi05 提供 SO-101 机械臂上 RL Token 的完整实现流程：采集示教数据、微调 π0.5、训练
RL Token、训练关键阶段分类器、真机在线强化学习，以及推理与部署。文中的命令均在一台 SO-101
工位（两条机械臂、两路相机）上验证过，参数可以直接使用，路径需替换为实际路径。

> [!WARNING]
> 从步骤 4 开始脚本会驱动真实机械臂。首次运行应先执行 dry run 预检，再用 `vla_only` 验证，
> 整个过程中操作者应始终将手放在急停开关上。

## 流程

```mermaid
flowchart TD
    S0["步骤 0 · 硬件准备：机械臂、相机、标定"] --> S1["步骤 1 · 采集示教数据（lerobot-record）"]
    S1 --> S2["步骤 2 · 微调 VLA π0.5（lerobot-train）"]
    S2 --> S3["步骤 3 · 训练 RL Token（预计算 embedding → 编解码器）"]
    S3 --> S4["步骤 4 · 训练关键阶段分类器（标注 → z_rl → 训练）"]
    S4 --> S5["步骤 5 · 真机在线 RL（Actor + Critic，train_rlt_stage2_pi05.py）"]
    S5 --> S6["步骤 6 · 推理与部署（infer_rlt_stage2_pi05.py）"]
```

整个流程严格串行，每一步只依赖上一步的产物。步骤 4 是唯一可选的环节，如果不训练分类器，可以在
运行时用 `c` 键手动切换关键阶段。

## 安装

```bash
git clone <本仓库> && cd RL-Token-Pi05
conda create -n rlt python=3.12 -y && conda activate rlt
pip install -e ./lerobot        # 内置的 lerobot fork，含 pi05_rlt
pytest                          # 仅需 CPU
```

如果不希望安装到环境中，也可以直接使用仓库内的源码：`export PYTHONPATH=$PWD/lerobot/src:$PYTHONPATH`。
环境中存在代理时 Hugging Face 下载会失败，而本项目所有资源都可以使用本地文件，因此建议在开始
工作前先执行 `unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy`。

## 步骤 0：硬件准备

工位由两条 SO-101 机械臂和两路相机组成，其中从臂负责执行动作，主臂由操作者手动引导。任务用到的
打印件模型放在 `assets/stl/`。串口通常枚举为 `/dev/ttyACM0` 与 `/dev/ttyACM1`，但重启后可能
互换，因此需要先确认：

```bash
lerobot-find-port
python so101_tools/check_follower_motors.py /dev/ttyACM0   # 从臂电机自检，会指出断点位置
sudo bash so101_tools/fix_so101_permission.sh              # 串口报 Errno 13 时使用
lerobot-find-cameras opencv
bash so101_tools/run_so101_teleop.sh                       # 遥操作试跑，同时清理占用端口的残留进程
```

相机配置建议为每路添加 `"fourcc": "MJPG"` 与 `"warmup_s": 3`。使用默认参数时，相机可能出现能够
打开却读不到画面帧的情况。相机编号在重启后可能发生变化，若需要固定设备，应使用
`/dev/v4l/by-path/` 下的设备路径。

每条机械臂需要标定一次，机械结构重新装配后应重新标定：

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=so101_leader
```

## 步骤 1：采集示教数据

正式录制前，建议先用遥操作确认动作方向与关节单位，确认无误后再开始采集：

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower \
  --robot.cameras='{"top": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG", "warmup_s": 3}, "wrist": {"type": "opencv", "index_or_path": 3, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG", "warmup_s": 3}}' \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=so101_leader \
  --dataset.repo_id=insert_bolt \
  --dataset.root=/path/to/lerobot/insert_bolt \
  --dataset.single_task="Insert the bolt into the hole" \
  --dataset.num_episodes=50 --dataset.episode_time_s=60 --dataset.reset_time_s=10 \
  --dataset.fps=30 --dataset.push_to_hub=false --display_data=true
```

采集时需要注意以下几点。任务文本在后续步骤中会反复使用，必须保持一致；除启用
`manual_positioning` 外，建议每集保持相同的起始位姿；对于单一任务，30 至 50 条质量良好的示教
数据通常已足够完成微调。数据集必须包含 `meta/stats.json`，在线强化学习依赖它进行归一化，缺少
该文件将无法启动。如果场景发生变化，例如更换了工具，应补充采集数据后再行合并。

## 步骤 2：微调 VLA

π0.5 是视觉-语言-动作模型，输入为相机画面、本体状态与语言指令，输出为动作序列。本步骤使用
lerobot 自带的训练器：

```bash
lerobot-train \
  --dataset.repo_id=insert_bolt --dataset.root=/path/to/lerobot/insert_bolt \
  --dataset.video_backend=pyav \
  --policy.type=pi05 --policy.pretrained_path=/path/to/pi05_base \
  --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \
  --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
  --policy.device=cuda --policy.optimizer_lr=5e-4 \
  --policy.scheduler_warmup_steps=1000 --policy.scheduler_decay_steps=30000 \
  --policy.scheduler_decay_lr=1e-5 \
  --output_dir=outputs/pi05_bolt --job_name=pi05_bolt \
  --batch_size=1 --num_workers=8 --steps=30000 --log_freq=10 \
  --save_checkpoint=true --save_freq=1000 \
  --policy.push_to_hub=false --wandb.enable=false
```

正式训练前，建议先将 `--steps` 设为 3 试跑一次，通常只需几分钟即可发现权重键名或显存方面的
问题。多卡训练可以用 `torchrun` 包装。学习率必须通过 `--policy.optimizer_lr` 设置；如果写成
`--optimizer.lr`，训练器会忽略该参数，修改不会生效。

每个保存点对应一个目录，后续步骤使用其中的 `pretrained_model/`：

```text
outputs/pi05_bolt/checkpoints/030000/pretrained_model/
├── config.json
├── model.safetensors
├── policy_preprocessor.json / policy_postprocessor.json   # 归一化状态
├── tokenizer/
└── train_config.json
```

VLA 只需能够接近目标区域即可，最后几毫米由在线强化学习修正；如果策略连目标附近都无法到达，
后续训练也无法弥补。

## 步骤 3：训练 RL Token

RL Token 将冻结 VLA 的内部表示压缩为一个 2048 维向量 `z_rl`，在线强化学习以它作为状态表示。
首先计算 embedding：

```bash
python scripts/precompute_pi05_embeddings.py \
  --pi05_path /path/to/pi05_bolt/checkpoints/030000/pretrained_model \
  --tokenizer_path /path/to/paligemma_tokenizer \
  --dataset_repo_id /path/to/lerobot/insert_bolt \
  --output_dir outputs/pi05_bolt_embeddings \
  --camera_map '{"top": "observation.images.top", "wrist": "observation.images.wrist"}' \
  --batch_size 16 --num_workers 4 --device cuda
```

产物包括 `prefix_out.mmap`、`prefix_mask.mmap` 与 `meta.json`。单卡处理 5 万帧约需 80 分钟；
由于该步骤可以按 episode 切分，使用 `--shard_index/--shard_count` 在多卡上并行执行可显著缩短
耗时，之后将各目录以逗号拼接后传给训练器。

```bash
python scripts/train_rlt_stage1_pi05.py \
  --precomputed_path outputs/pi05_bolt_embeddings \
  --output_dir outputs/rlt_stage1_bolt \
  --steps 30000 --batch_size 64 --lr 1e-4 --warmup_steps 200 \
  --save_every 1000 --log_every 50 --num_workers 4 \
  --device cuda --gpus 0 --use_amp
```

`--gpus` 只接受单块显卡，因为 encoder 与 decoder 不拆分到多卡。VLA、embedding、该 checkpoint
以及后续的在线强化学习配置必须来自同一个 VLA；混用会使 `z_rl` 的分布发生偏移，导致策略性能
下降且不会报错。

训练产物为单文件 `outputs/rlt_stage1_bolt/best_checkpoint.pt`，后续脚本均读取该文件。

## 步骤 4：训练关键阶段分类器

在线强化学习只从关键阶段学习，也就是真正决定任务成败的那段动作，例如最终对准、插入以及闭合
夹爪。运行时可以按 `c` 键手动切换，分类器则利用 `z_rl` 与本体状态自动完成这一判断。由于分类器
以 `z_rl` 为输入，必须安排在步骤 3 之后。

`scripts/annotate_dataset.py` 会把一集的多个相机画面并排播放，并记录标出的区间。结果在每
一集结束后写入 `--annotations_out`，退出时还会再写一次，因此随时可以中断再继续，文件里
已经有的集会自动跳过。

<p align="center">
  <img src="assets/annotate_tool_gui.png" alt="标注工具：两块相机画面与已标出的区间" width="780">
</p>

<p align="center"><i>每个相机一块画面，画面下方是相机名。状态栏显示当前集数、帧号以及已经标好的区间；区间闭合
后，播放头落在其中时画面会被红框框出。一集播完自动进入下一集。</i></p>

| 按键 | 作用 |
| --- | --- |
| `s` | 在当前帧开始一个区间 |
| `e` | 在当前帧结束区间 |
| `u` | 撤销上一个区间 |
| `space` | 暂停 / 继续 |
| `←`、`→` | 后退 / 前进一帧 |
| `Shift` + `←`、`→` | 后退 / 前进 30 帧 |
| `n` | 下一集 |
| `q` | 保存并退出 |

没有显示器时会退回胶片模式：把每一集渲染成带帧号的缩略图拼图，在终端里填写区间。

```bash
# 1. 标注关键区间
python scripts/annotate_dataset.py \
  --dataset_path /path/to/lerobot/insert_bolt \
  --annotations_out annotations_bolt.json

# 2. 计算采样帧的 z_rl
python scripts/precompute_critical_zrl.py \
  --pi05_path /path/to/pi05_bolt/checkpoints/030000/pretrained_model \
  --tokenizer_path /path/to/paligemma_tokenizer \
  --rlt_checkpoint outputs/rlt_stage1_bolt/best_checkpoint.pt \
  --dataset_path /path/to/lerobot/insert_bolt \
  --annotations annotations_bolt.json \
  --output outputs/critical_zrl_bolt.npz \
  --pos_per_interval 40 --neg_per_episode 120 --batch_size 8 --device cuda

# 3. 训练分类器
python scripts/train_critical_classifier.py \
  --precomputed_zrl outputs/critical_zrl_bolt.npz \
  --output outputs/critical_classifier_bolt.pt \
  --epochs 60 --batch_size 64 --lr 1e-3 \
  --hidden_dim 256 --num_layers 2 --val_episodes 6 --device cuda

# 4. 查看指标并确定阈值
python scripts/evaluate_critical_classifier.py \
  --precomputed_zrl outputs/critical_zrl_bolt.npz \
  --classifier outputs/critical_classifier_bolt.pt --val_episodes 6 \
  --threshold_on 0.75 --threshold_off 0.45 --smooth_steps 4 --device cuda

# 5. 画出可视化核验图
python scripts/visualize_critical_classifier.py \
  --precomputed_zrl outputs/critical_zrl_bolt.npz \
  --classifier outputs/critical_classifier_bolt.pt \
  --out_dir outputs/critical_vis --mode both \
  --dataset /path/to/lerobot/insert_bolt --annotations annotations_bolt.json \
  --episodes 0-11 --cam top --device cuda
```

核验图会写出两类文件：`prob_timeline.png` 每个回合一条概率曲线并标出人工标注区间，`sheet_XX.png` 是各
区间边界帧的拼图，每帧下方标注分类概率。下图即其中一张拼图，裁切保留前两行。

<p align="center">
  <img src="assets/critical_phase_classifier_check.jpg" alt="关键阶段分类器核验" width="780">
</p>

<p align="center"><i>分类器核验图：每行一个回合，红框是人工标注的关键阶段之外的一帧，绿框是阶段内的三帧，每帧下方
是该帧的分类概率。这里展示 6 个核验回合中的 2 行；整批 30 个区间的统计为区间内平均概率
0.92~0.99、区间外 0.002~0.09。</i></p>

标注质量决定了分类器的性能上限，因此各集的标注标准应保持一致。如果开关切换的滞后明显，应重新
标注，单纯调低阈值只是把问题转移到其他位置。分类器效果不理想不会带来风险，因为手动按下的 `c`
键始终优先。

## 步骤 5：真机在线 RL

配置文件模板为 `configs/stage2_pi05.json`，其中带有注释，通常只需修改路径：

```bash
python scripts/train_rlt_stage2_pi05.py --config configs/stage2_pi05.json
```

建议按以下顺序推进：先执行 dry run，该模式在模板中默认开启，只检查路径、权重、相机、归一化
统计与配套关系，不访问硬件；随后在真机上运行 `vla_only`，确认相机画面、动作方向与关节单位
正确；然后启用 Actor 运行少量回合；最后开始正式训练。上述开关只能在命令行打开，若要连接真机，
需要在配置中将 `"dry_run"` 改为 `false`，命令行不支持 `--dry_run false` 的写法。新实验需要空的
输出目录，续训同样需要新的空目录，并通过 `--resume_from` 指向原有 checkpoint。

一个回合内的流程如下：冻结的 π0.5 先规划出 10 步参考动作，RL Token 编码器计算 `z_rl`，Actor
（实际输出动作的网络）在关键阶段接管并输出动作块，从臂执行之后由操作者用 y/n 判定成败并写入
回放缓冲，Critic（评估动作价值的网络）在后台异步更新。前 `warmup_episodes` 个回合由纯 VLA
执行，只有关键阶段的数据进入回放缓冲。

> [!TIP]
> 预热阶段非常关键：它用 VLA 自己在关键阶段的表现把回放缓冲填起来，Critic 的第一批学习信号
> 来自这里，Actor 也才能从一个像样的策略出发，而不是从随机的动作开始。实践上 `warmup_episodes`
> **建议大于 15**；模板默认的 5 只够把流程跑起来，样本量太小，最初的几次 Actor 更新会来自一个
> 几乎空的缓冲。

| 按键 | 作用 |
|---|---|
| `c` | 切换关键阶段，按一次开始记录回放，再按一次停止。每个回合会重置 |
| `a` | 切换 Actor 执行，状态跨回合保持 |
| `p` | 回合前摆位（需启用 `manual_positioning`），摆位过程不会写入回放 |
| `h` / `s` | 人工接管 / 立即交还策略 |
| `f` / `g` | 提前结束本回合 / 丢弃本回合 |
| `e` / `q` | 急停 / 退出 |
| `y` / `n` | 弹窗提示时判定成功 / 失败 |

一次带人工介入的回合大致如下：回合开始后按一次 `a` 启用 Actor（该状态跨回合保持），关键阶段处于
激活状态时 Actor 才会接管，关键阶段由分类器（`--auto_critical`）或手动按 `c` 决定；策略运行期间
主臂跟随从臂，两条臂一起运动。发现策略即将失败时按 `h` 切换到人工控制，从臂从当前位姿开始跟随
主臂，因此修正动作不会出现跳变；再按一次 `h` 或按 `s` 把控制交还策略；按 `f` 在当前动作块结束后
结束本回合，随后在 y/n 提示中判定成败，下一回合随之开始。介入过程会作为修正动作保留，成败在回合
结束的提示中判定，而不是在交还控制时判定。

若希望关键阶段自动进入和退出，可以加入
`--auto_critical --critical_classifier outputs/critical_classifier_bolt.pt`，此时手动按下的 `c` 键
仍然优先。

控制台只输出回合计数与必要提示，完整日志写入 `<output_dir>/train.log`，每个回合的指标记录在
`rl_metrics.log`，可以使用 `python scripts/plot_rl_metrics.py <日志路径> --live --port 8000`
查看训练曲线。
精确的 Critic loss、Actor loss 和 Actor 采样 Q 另写入 `<output_dir>/rl_losses.log`，可用同一脚本打开该文件生成损失曲线。

<p align="center">
  <img src="assets/rl_training_curves.png" alt="在线 RL 训练曲线" width="780">
</p>

<p align="center"><i>`plot_rl_metrics.py` 渲染的一次在线 RL 运行曲线。</i></p>

<p align="center">
  <img src="assets/rl_token_epochs.gif" alt="RL Token 训练 5 轮与 50 轮的对照" width="700">
</p>

<p align="center"><i>RL Token 训练到 5 轮与 50 轮时在同一任务上的对照：随着训练推进，动作逐渐收敛。</i></p>

## 步骤 6：推理与部署

推理脚本在真机上运行训练好的策略，过程中不进行学习，也不会更新权重。它的配置文件是
`configs/infer_pi05.json`，需要填写的内容包括各路径、相机与串口，以及要加载的 actor 存档
（`resume_from` 键，指向步骤 5 产出的 `final_checkpoint.pt`）。配置填好后直接启动即可：

```bash
python scripts/infer_rlt_stage2_pi05.py --config configs/infer_pi05.json
```

首次运行请保持 `"dry_run": true`，脚本只做检查而不访问硬件；确认无误后改为 `false`。只提供 π0.5
基座并不构成一个训练好的策略，因此 `resume_from` 为空时脚本会直接报错退出。

常用参数包括：`--vla_only` 仅运行 VLA，用于首次检查；`--record_dataset` 将 rollout 录制为可以
直接用于再训练的数据集；`--leader_mirror` 把从臂位姿镜像到主臂；`--inference_score` 在每个回合
结束后询问 y/n 以统计成功率；`--save_debug_obs` 导出逐步的 state/target/sent CSV 文件。启动前
脚本会完成相机预检与配套关系校验，任一项不通过即停止运行。

## 配置项

训练用的配置文件是 `configs/stage2_pi05.json`，推理用的是 `configs/infer_pi05.json`，两者都带
注释。训练配置中通常需要修改的配置项如下：

| 键 | 说明 |
|---|---|
| `pi05_path` | 微调后的 π0.5 目录，即 `pretrained_model/` |
| `rlt_checkpoint` | 步骤 3 生成的单文件 `.pt` |
| `tokenizer_path` | tokenizer 目录，离线机器应使用本地副本 |
| `stats_path` | 数据集的 `meta/stats.json`，留空则从 checkpoint 中读取 |
| `task` | 语言指令，需与数据集中的 `single_task` 一致 |
| `follower_port`、`leader_port` | 两条机械臂的串口 |
| `cameras`、`camera_map` | 相机设备及其对应的模型输入 |
| `control_fps`、`steps_per_episode`、`max_episodes` | 回合节奏 |
| `warmup_episodes` | Actor 启动前的纯 VLA 预热回合数，建议大于 15 |
| `replay_warmup_ratio` | 每个在线 replay batch 中 warmup/offline 数据的比例，默认 `0.3`，其余为 online/HIL |
| `replay_online_decay_steps` | online/HIL 数据的插入年龄衰减常数，默认 `2000` 条 transition |
| `replay_online_uniform_floor` | online/HIL 池的均匀概率下限，默认 `0.2` |
| `output_dir` | 输出目录，新实验需要空目录 |
| `resume_from` | 续训时指向既有 checkpoint，留空表示新训练（仅训练配置） |
| `dry_run` | 首次运行必须为 `true` |

其余配置项，包括预热回合数、Actor 启用门槛、自动关键阶段的阈值与平滑参数、HIL 相关设置以及
网络结构等，在模板中均有注释说明。部署脚本的每个参数、默认值以及需要修改的情形见
[CONFIG.zh-CN.md](https://github.com/Zhengsw03/RL-Token-Pi05-open/blob/e3c1f4517108fdff66d93b68bae3a16fc1cd5dd3/CONFIG.zh-CN.md)。

## 出问题怎么办

遇到问题时可以先对照下表：

| 现象 | 处理方式 |
|---|---|
| 串口报 `PermissionError` | 执行 `sudo bash so101_tools/fix_so101_permission.sh` |
| 某个电机没有响应 | 执行 `python so101_tools/check_follower_motors.py /dev/ttyACM0` |
| 相机能打开但读不到帧 | 添加 `"fourcc": "MJPG"` 与 `"warmup_s": 3` |
| 修改学习率后不生效 | 改用 `--policy.optimizer_lr` |
| 提示 checkpoint 与 VLA / RL Token / stats 不配套 | 三者必须同源，需按同一份 VLA 重新训练 |
| 分片缓存提示维度不一致 | 检查各分片 `meta.json` 中的 `seq_len` 与 `vlm_hidden_dim` |
| 按 `f` 没有反应 | y/n 弹窗正在等待输入，先按 y/n/r/q |

运行测试使用 `pytest`（仅需 CPU），在线强化学习的契约测试可用
`pytest tests/test_train_rlt_stage2_pi05.py -v`。

## 致谢

本工作在中兴通讯（ZTE）提供的场地与设备支持下完成。

感谢 Physical Intelligence 团队开源 π0.5（本仓库微调的 VLA），以及 LeRobot 项目提供的采集、
训练与部署框架。

感谢中兴通讯工程师、本仓库贡献者 [@Teddy-Liao](https://github.com/Teddy-Liao) 与
[@TomwKang](https://github.com/TomwKang) 在硬件搭建与实验过程中给予的帮助。

<p align="center">
  <img src="assets/zte_logo_zh.png" alt="中兴通讯 ZTE" width="200">
</p>

### 招聘

中兴通讯机器人团队（ZTE Robotics）从事具身智能与机器人操作方向的研究，正在招聘 manipulation、
任务规划、运控等岗位。

- 中兴通讯机器人团队工程师、本仓库协作者：[@Teddy-Liao](https://github.com/Teddy-Liao)
- 本仓库作者：[@Zhengsw03](https://github.com/Zhengsw03)

如果这个仓库对你有帮助，欢迎点个 star。

<a id="贡献指南"></a>

## 🤝 贡献指南

欢迎 Issue 与 PR：问题反馈、文档订正、示例、新任务复现与硬件经验都可以。报告需要包含哪些信息、
PR 有哪些检查项见 [CONTRIBUTING.md](https://github.com/Zhengsw03/RL-Token-Pi05-open/blob/e3c1f4517108fdff66d93b68bae3a16fc1cd5dd3/CONTRIBUTING.md)；中英文两份 README 保持同步，改动一处时另一处
一并更新。贡献按 Apache License 2.0 发布。

共同推进机器人进步。

## 许可证

本项目采用 Apache License 2.0，详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。`lerobot/` 目录是
[huggingface/lerobot](https://github.com/huggingface/lerobot) 的修改副本，按同一许可证分发。

方法来自 *RL Token: Bootstrapping Online RL with Vision-Language-Action Models*
（arXiv:2604.23073），项目主页为 <https://pi.website/research/rlt>。
