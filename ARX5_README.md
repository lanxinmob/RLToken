# 本机 ARX5 鼠标任务 RLT

本仓库负责 prefix、Stage 1、关键阶段分类器、Stage 2 在线 RL 与 RLT 评估。
π0.5 基座、SFT、单臂 VLA 推理、示范数据及 VLA 权重由 [`../pi05`](../pi05/README.md) 管理；
RLT 配置只读取它们。所有 RLT 改动直接提交到本仓库，不再修改 `pi05/.deps` 中的训练脚本。

```bash
cd "$(git rev-parse --show-toplevel)"
./rlt check --gpu
./rlt stage2-check --auto-critical
./rlt stage2 --auto-critical --episodes 5
```

当前环境 `.venv` 已准备好；新建环境运行 `bash setup_arx.sh`，依赖固定在 `arx-lock.txt`。
模型与编码器来源及硬件配置集中在 `rlt_stage2.json`；SDK Python 位于 `config.json`。
当前计划见 [RLT_REPRODUCTION_PLAN.md](RLT_REPRODUCTION_PLAN.md)，交接现状见 [HANDOFF.md](HANDOFF.md)，
历史硬件证据见 [HARDWARE_LINK_REPORT.md](HARDWARE_LINK_REPORT.md)。

已有 50 条数据、28,000 步 VLA 和 30,000 步 Stage 1，无需重复训练。需要重新提取/训练时：

```bash
./rlt precompute --output outputs/rlt/embeddings_new
./rlt stage1 --embeddings outputs/rlt/embeddings_new --output outputs/rlt/stage1_new --steps 30000
```

大文件迁移使用同盘移动。原日志、checkpoint/cache 内部的训练来源保持原样，历史路径映射保存在
`outputs/migration_20261029/path_mapping.json`。不要把旧日志内的路径当作当前启动目录。

## ARX Stage 2

模型/learner 使用 `.venv`；`rlt_stage2_hardware.py` 使用 `config.json` 中的 SDK Python。
控制依据为 `../pi05/pi05_hardware.py`，主从拖动与夹爪映射依据
`../arx5-sdk/python/examples/leader_follower_teleop.py`。配置集中在
[`rlt_stage2.json`](rlt_stage2.json)，启动前结束其他 `../pi05/pi05 infer`、SDK 遥操和 openpi 硬件进程，
并核对执行臂标定与实际工作区；若恢复双臂部署，再额外核对主臂。

```bash
cd "$(git rev-parse --show-toplevel)"
./rlt stage2-check --auto-critical
# 以下会连接当前配置的执行臂和腕相机；先进行 5 个现场验收回合：
./rlt stage2 --auto-critical --episodes 5
```

`stage2` 默认在桌面打开 `RLT ARX5 | cam_wrist` 实时相机窗口；按窗口中的 `q` 或 Esc
只关闭预览，终端中的 `q` 仍然是正常退出训练。无桌面环境时预览会自动跳过，也可显式使用
`./rlt stage2 --no-preview ...`。

每回合默认执行上限为 900 个控制步（30 Hz 下约 30 秒），也可用
`--steps-per-episode` 覆盖。训练时一旦进入关键阶段，片段会持续到人工按 `c` 关闭，或按 `f` 结束并评价，
即使达到 900 步也会继续；`c` 关闭后交回 VLA，达到上限或按 `f` 才评价。
尚未触发关键阶段时仍按原上限结束。
`f` 后用 `y/n/d` 标记成功、失败或丢弃；`q` 退出整个实验。

鼠标任务的关键阶段分类器已于 2026-10-22 完成训练，权重为
`outputs/rlt/critical_mouse_20261022_181349/critical_classifier.pt`。
使用 50 条、14,709 帧人工区间标注的示范：第 6–49 集训练，第 0–5 集验证。
冻结 28,000 步 VLA 和 30,000 步 Stage 1 编码器，输入 `2048D z_rl + 7D 原始状态`，
训练 `2055 → 256 → 2` MLP（Adam，lr=1e-3，交叉熵 + 0.05 时间变化惩罚）。
每次更新输入一条完整回合序列，训练 60 轮，保存验证集最优的第 10 轮。
原始分类验证准确率为 95.56%；每 10 帧更新检测器时 F1=89.66%，区间 IoU=81.15%，
进入平均晚 0.606 秒。完整命令、曲线及验证限制见
[分类器训练报告](outputs/rlt/critical_mouse_20261022_181349/README.md)。这些是离线结果。

2026-10-29 改进实验：仅在原 44 条训练轨迹上做按回合交叉验证，保留原权重、调整检测参数，
在原 6 条验证轨迹上将每 10 帧检测的 F1 从 89.66% 提升到 93.16%、召回率从 85.89% 提升到
92.70%，平均进入延迟从 0.606 秒降到 0.272 秒，误报帧从 46 增至 51。
正则化重训未提高逐帧准确率，因此权重与默认检测配置均保留；候选配置、命令及限制见
[改进实验报告](outputs/rlt/critical_review_20261029_204952/README.md)。仍需新回合独立验证和真机短试。

`--auto-critical` 使用配置中的分类器和已验证检测参数（ON=0.75、OFF=0.45、
smooth_steps=4、min_on_chunks=0）；需要换路径时使用 `--critical-classifier`。
只读检查也会加载分类器并核对 `2048D/7D` 维度，随后退出，不连接硬件。
在线分类器输入原始关节弧度和夹爪米数，再使用权重内的训练集均值/标准差；
Actor/Critic 仍使用自己的分位数归一化。默认前 60 个控制步不自动检测，
每个动作窗口用同一份观测的 token 和实测状态检测一次，每回合重置检测历史。
训练时分类器只触发关键阶段进入，之后不因概率下降退出。`c` 可手动截断片段，
在当前 10 步动作窗口结束时停止采集并锚定终点；自动检测不会把手动关闭的片段重新打开。
`c` 只关闭关键阶段，不直接结束回合；随后由 VLA 执行，`f` 结束并评价。
这些在线门控及推理停顿不包含在上述离线回放指标中。
省略 `--auto-critical` 则保留原来的手动 `c` 标记方式。

历史双臂验收时从臂为 `X5_umi/can1`、主臂为 `X5/can0`；当前本机只有执行臂
`X5_umi/can0`，默认配置改为显式单臂 HIL。夹爪全开读数为 `-3.45369`，重力倍率为
`1.0`。默认关节安全限速为 `3.6 rad/s`（每个 30 Hz 目标最多变化 `0.12 rad`）、夹爪为 `0.01 m/s`。
硬件进程以 100 Hz 检查反馈，策略目标以 30 Hz 直发；SDK 按其 2 ms 控制周期插值。
自主目标不再经过额外 EMA，连续性由 RTC (`guided`, execution horizon=10) 处理。单臂人工接管时切换同一只执行臂为低刚度拖动，直接读取
实测状态，不再向同一 CAN 总线回发人工目标；恢复策略后从新的实测边界观测重新规划。
Actor-only 的相对目标限幅已禁用；VLA 与 Actor 都使用同一份执行契约，按执行边界的
关节/夹爪速度和 SDK 工作空间限幅后再写入 replay。跟踪误差阈值只用于故障检测，不参与目标整形。不要在配置中设置
`actor_max_relative_target`，否则启动会拒绝该配置。
Stage 2 按论文保留每窗口 10 个连续动作、replay 步距 2：整段关键阶段先采集，再切成重叠窗口训练，
每条样本的动作形状为 `10×7`。VLA 每次仍预测 50 步，每个窗口使用新计划的前 10 步。
`../pi05` 的纯 VLA 参数 `--replan-steps 25` 不用于这个训练窗口。

当前 `rlt_stage2.json` 使用 `rtc_enabled=true`、`rtc_mode=guided`、`rtc_execution_horizon=10`、
`async_inference=false`、`replan_every_window=true`，
动作推理在窗口边界同步完成：执行满 10 步后采集一份图像与实测状态，用这份观测生成
VLA 参考、`z_rl` 和 Actor 动作，再执行下一窗口的 10 步。推理期间硬件维持最后接受的目标，
不会继续提交旧动作块的后续目标；等待不增加策略步数或 replay 动作。人工接管结束后从新观测重算。

learner 继续在独立线程中采样 replay、更新 Actor/Critic，与 rollout 采集异步运行。
不足 10 步的回合尾样本使用 `valid_mask`：BC 只计算已执行位置，Actor 的 Q 输入也将未执行位置
置零，与 replay 的补齐方式一致。更新次数只能说明 learner 在工作；判断改善应单独统计无 HIL
回合的成功率，并检查 `debug_obs/episode_*/steps.csv` 和 `hardware_actions.jsonl` 中请求目标与
实际接受目标的偏差。2026-10-31 的在线检查及历史权重异常见
[检查报告](outputs/rlt/online_audit_20261031/README.md)。修复代码不会自动修复旧 checkpoint 的异常输出。
窗口内目标按 30 Hz 下发，跨窗口还需等待同步推理和通信，实际平均动作频率需包含这些等待。
Actor 探索噪声按 chunk 采样一个 7D 向量并广播到 10 步；Actor 输出在归一化空间钳位到 ±1.5。
Actor 只有在 learner 最近一次 batch 均值 `Q(actor)-Q(reference)` 严格大于
`actor_gate_advantage` 时才接管，否则保持 VLA；指标日志中的 `referenceQ` 和 `actorAdv`
可直接判断闸门状态。BC 权重从 warmup 的 10 线性退火到 2.5，Q 权重从 1 退火到 4。
实验性的动作预取会把较早观测生成的 VLA 参考交给当前状态的 Actor，不用于当前论文对照配置。

Stage 2 的 SigLIP 视觉注意力使用 `eager`：本机默认 SDPA 在固定图像的串行对照中存在波动，
切换后重复特征的最大差值为 0；不修改权重、精度或全局系统配置。
历史性能测量及动作预取试验见 [HANDOFF.md](HANDOFF.md)。
2026-10-30 恢复同步配置后，51 项原有单测及根入口只读预检通过，未打开 CAN 或实时相机。
实际真机频率及 learner 竞争仍需现场测量。

启动后执行臂保持当前实测姿态。交互式终端中的命令为单键即时生效，不需要回车：

| 命令 | 行为 |
| --- | --- |
| `p`，再 `p` | 执行臂进入重力补偿，可直接拖动和开合夹爪；再按一次保持当前姿态并开始本回合，摆位数据不进入 replay |
| `c` | 手动开启/截断关键训练阶段；关闭后停止该片段采集并交回 VLA，终点取关闭边界；每回合开始时重置 |
| `h` | 人工接管/交还策略；单臂模式将执行臂切到低刚度拖动，双臂模式使用主臂拖动 |
| `s` | 结束人工接管并恢复策略 |
| `a` | 禁止/重新允许 Actor 执行；`--auto-critical` 启动时默认允许，实际执行仍受关键阶段、warmup 和更新次数限制 |
| `f` | 提前结束回合，保留已采集数据并询问成败 |
| `y` / `n` / `d` | 在回合末标记成功/失败/丢弃 |
| `e` / `q` | 急停或正常退出；`q` 会先缓慢回到启动时的 home pose，再保存训练产物 |

等待 `p` 或回合成败标记时，执行臂保持当前位置。第一次 `p` 后执行臂关节和夹爪为
零位置刚度、低阻尼，保留重力补偿；直接拖动执行臂，摆位期间不执行策略。
第二次 `p` 从实测姿态恢复位置保持后启动策略，不使用摆位前的旧目标。
摆位和人工接管不受策略观测超时约束，切回策略后恢复该检查。相机、执行臂反馈和通信保护
全程生效；摆位循环、接管和策略执行均受 2 秒中断超时保护。

接管时保持当前执行臂姿态并清除旧目标，确认新鲜反馈后切换为低刚度拖动。按下 `h` 到
真正开始拖动之间的静止/交接样本只用于刷新反馈序号，不增加 `episode_step_counter`、
`StepRecord` 或 replay；默认要求关节变化达到 `0.005 rad` 或夹爪变化达到 `1 mm` 才开始
记录。单臂模式不做
主从夹爪匹配，也不向 CAN 回发拖动目标；动作日志记录的是实际测得的关节弧度和夹爪米数。
若恢复双臂模式，才使用原有的主从夹爪匹配和 86 mm 从臂上限。
回合间保持并人工摆位，自动回零关闭。急停、通信/推理超时、反馈过期或断图会停止执行臂、
锁定故障，结束本次会话；终端按 `q` 正常退出时会用启动时记录的 home pose
缓慢回位，再进入阻尼；急停或硬件故障不会追加回位动作。

Stage 2 的 `c` 是关键训练阶段开关，`s` 是交还策略，不是 `../pi05/pi05 infer` 中的开始与暂停保存。
分类器仅触发进入；用 `c` 截断时终点取关闭边界，否则取人工 `f` 结束时的观测。
片段奖励仍由回合末 `y/n` 提供。
VLA、Stage 1 编码器及分类器冻结，只在线更新 Actor 和双 Critic。
默认前 20 回合仅采集 VLA 和人工纠正，之后 replay 至少有 256 条样本才开始更新。
使用 `--auto-critical` 时默认允许 Actor，累计满足 1,000 次更新后在检测到关键阶段的第一个窗口接管，
不需要手动按 `a`，`actor_critical_delay_steps=0`。`h/s` 仍可人工接管/交还，片段持续到手动 `c` 关闭或 `f`。
Actor 训练时以 50% 概率将 reference 输入置零，并保留 BC 正则（warmup 权重 10，在线退火至 2.5）；
BC 目标保留完整 reference，执行时始终提供完整 reference。HIL 动作作为对应的 BC 目标。
关键阶段之外仍由 VLA 执行；按 `a` 禁止后会跨回合保持禁止，直到再次按 `a`。
省略 `--auto-critical` 时仍需手动 `c` 标记关键阶段、`a` 允许 Actor。
5 回合检查仅用于现场验收，不应期待产生 RL 更新。
先检查镜像、`h/s` 衔接、成败标签、replay 和停止行为，再继续正式训练。
相同模型、统计和硬件配置下可以从刚才的 checkpoint 续训：

```bash
./rlt stage2 --auto-critical --episodes 200 \
  --resume-from outputs/rlt/stage2_arx_YYYYMMDD_HHMMSS/final_checkpoint.pt
```

每次启动使用新输出目录 `outputs/rlt/stage2_arx_<时间>/`，也可指定 `--output`。
`--episodes 200` 指训练到第 200 回合，恢复时包含已完成回合；不恢复时去掉 `--resume-from`。
修改关键阶段接管/终止规则后可沿用现有 checkpoint：保留 Actor/Critic、优化器、累计更新数和
checkpoint 边界内的 replay。先结束并评价当前回合，再按 `q`，等保存 `final_checkpoint.pt` 后用
新命令恢复；运行中的进程不会自动加载代码变更。若没有 final 文件，改用最近已保存的
`checkpoint_epN.pt`，从第 N+1 回合继续。旧 replay 保留其原有片段边界，新回合使用新规则。
目录包含 `train.log`、`rl_metrics.log`、replay journal、训练 checkpoint、`hardware_info.json`
及 `hardware_actions.jsonl`；动作日志和 replay 使用 SDK 实际接受的限速目标。
换 VLA 时须重新准备配套 prefix/encoder，不能只换推理 checkpoint。

若启动报相机 `near-black/no-signal`，先检查 Vin 1 腕相机的供电、输入和线缆。
硬件进程会在 `cameras.cam_wrist.warmup_s`（默认 3 秒）内等待连续有效新帧，
通过后才连接执行臂。SDK 初始化会暂时阻塞同进程的相机采集线程，因此执行臂保持当前姿态后，
会再等待初始化之后收到的连续有效新帧，随后报告就绪；每次等待最多 `warmup_s`，
`startup_timeout_s` 必须覆盖两次等待。运行中相机帧龄上限仍为 `0.2s`，持续黑屏或断图仍会退出。
终端直接显示硬件日志末尾和完整路径。
可先单独检查相机（不连接 CAN），通过后再运行 `./rlt stage2 --episodes 5`：

```bash
cd ../arx5-sdk/python
${ARX_PYTHON:-python} -B examples/record_episodes.py --check-cameras --no-preview
cd "$(git rev-parse --show-toplevel)"
```

若启动报 `Emergency latched`，查看之前的 SDK 错误。旧配置将 `can1` 从臂当成主臂，
其 `-3.499084 rad` 反馈被按主臂标定换算成约 93.4 mm，触发
`Gripper position error: got 0.093 but should be in 0~0.088`。按从臂既有标定换算约为
87.1 mm，处于 SDK 的宽度加 5 mm 容差范围内。这些值是电机角度换算，不是实物宽度的独立测量。
若手动改变开度后仍报同一个值，退出所有控制程序，运行只读反馈检查：

```bash
./rlt gripper-check --seconds 60
```

检查只向当前配置的 CAN（单臂默认 `can0`）8 号电机发送达妙状态查询，不构造控制器；每秒显示
原始角度、按现有标定换算的毫米数和角度范围，原始数据保存在终端所示 JSONL。单臂模式下直接
开合执行臂核对读数；双臂模式再分别核对 `leader/can0`、`follower/can1`。开合时只有另一路变化则核对
实物接线与 CAN 映射；对应电机角度不变则继续查传动与编码器反馈；角度随开合变化但范围
不符则按 SDK 的[夹爪排查](../arx5-sdk/docs/hardware/troubleshooting.md#gripper)复核零点和全开标定。
查询命令正常退出仅表示收到反馈，不代表标定或真机控制验收通过。

2026-10-21 离线验证：真实权重 dry-run、17 项 ARX 控制检查、75 项上游回归检查通过；
真实 VLA/encoder 在替身机械臂上跑完 3 个模拟回合和 32 次 learner 更新；另以 2 个模拟回合
验证 Actor 执行、人工纠正、`10×7` replay，以及 checkpoint/replay 恢复。
临时测试、配置和模拟产物已删除。上述结果不代表实物动作验收或任务成功率。

2026-10-29 离线验证：`stage2-check --auto-critical` 成功加载真实 VLA、encoder 和分类器；
验证分类器在硬件连接前加载并拒绝维度不符、原始状态及录制输入一致、Actor 归一化独立、
手动默认和自动参数传递均通过；重算 1,826 帧验证准确率与训练报告一致。没有启动真机运动，
没有保留临时测试文件。已有完整真机会话未产生 RL 更新，最近一次因主臂夹爪负开度反馈退出，
仍需现场验收反馈、接管与 replay 后再开始正式训练。
