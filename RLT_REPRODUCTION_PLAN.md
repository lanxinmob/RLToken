> 2026-10-29 目录职责已调整：RLT 工作目录是 `<repo-root>`；π0.5 SFT/纯 VLA 在 `../pi05`。本文保留历史计划与命令；当前已验证入口、资产和操作以 [ARX5_README.md](ARX5_README.md) 为准。旧绝对路径映射见 `outputs/migration_20261029/path_mapping.json`。

# ARX 主从臂 RLT 复现操作计划

更新：2026-10-15。第三方双臂多任务部署已迁至 [openpi](../openpi/examples/arx5/README.md)，本目录仅用于 RLT 完整复现。
本轮按 2026-10-22 现场确认的 X5/can0 主臂、X5_umi/can1 从臂和现有鼠标任务，
完整执行上游的示教 → VLA → RL Token → 关键阶段 → 在线 RL → 部署流程。
当前完成环境配置和离线短链路检查；现场 Vin 1、左右臂反馈及基础人工主从遥操通过，左夹爪闭合零点待验证。
人工遥操使用 SDK `8de7590`，两臂重力倍率显式 1.0、目标速度上限 0.15 rad/s / 0.01 m/s，速度前馈和夹爪力反馈关闭。
主臂倍率 1.2 下曾自行漂移，1.0 不代表所有姿态已稳定；启动/跟随/暂停/退出结果见 [硬件报告](HARDWARE_LINK_REPORT.md)。
本文后续勾选项是执行计划，不表示已完成正式训练、RLT 在线控制或全部运动验收。

## 1. 固定目标与完成条件

任务指令固定为 `put the mouse on the mouse pad`。主臂负责示教/人工接管，从臂负责执行。
开始每轮前把鼠标放在记录过的工作区域，鼠标垫位置、腕相机安装和工具保持与示教一致。
初期按记录的起始姿态人工摆位，不自动回零。

本轮现场成功标准：鼠标完整落在鼠标垫内，夹爪释放后鼠标稳定；结束画面与人工 y/n 标签同时保存。
正式评估前固定稳定观察时长、允许的起始位置范围和每轮时间上限，所有比较使用同一标准。
失败、人工接管、放弃、通信故障分别记录；人工救回的回合不能计入纯策略自主成功率。

项目跑完需要同时达到：

- [x] 独立环境、SDK 导入、GPU 计算和真实数据合同通过。
- [x] 本地基座严格加载，真实样本能训练、提取 prefix、训练 RL Token、加载 7D Actor/Critic。
- [x] SDK 基础人工主从遥操、暂停/恢复及正常退出完成现场测试。
- [ ] 正式 VLA 微调完成，并在同一鼠标任务上通过基线真机评估。
- [ ] 正式 Stage 1 完成，VLA、缓存、tokenizer、归一化和 RL Token 来源一致。
- [ ] ARX 主从运行器的保持、镜像、人工接管、限幅、超时和停止通过实测。
- [ ] 用手动关键阶段完成在线 RL；再完成自动关键阶段分类器分支。
- [ ] 导出 checkpoint，重新加载、重启部署并完成独立成功率评估。

换用 ARX、单路腕相机和鼠标任务属于任务迁移。expert-only 微调属于训练变体；
以上流程完成也不等于复现原 SO-101 任务的论文性能。

## 2. 当前环境与产物

工作目录 `../pi05`。SDK 实际路径是 `../arx5-sdk`，SDK Python 是
`${ARX_PYTHON:-python}`。离线检查没有构造 SDK controller。

| 项目 | 当前事实 |
| --- | --- |
| RLT 源码 | `.deps/rlt`，固定 `cccf949c782266f3dab227ec4265a6360e0511e2` |
| RLT 环境 | `.venv-rlt`：Python 3.12.14、LeRobot 0.6.2、Torch 2.7.1+cu128、Transformers 5.5.4 |
| 依赖复现 | `rlt-requirements.txt` 输入，`rlt-lock.txt` 固定所有安装版本 |
| 本地源码修改 | `rlt-upstream.patch`；安装脚本幂等应用，禁止直接改完忽略补丁 |
| 原 π0.5 环境 | `.venv` 仅用于基座权重准备，旧8005部署服务已停止 |
| 原始示教 | `../arx5-sdk/python/data/demonstrations/episode_000004.hdf5`，真实成功记录 |
| LeRobot 数据 | `outputs/rlt/dataset`：50 集，14,709 帧，30 FPS，约 0.99 GiB |
| RLT π0.5 基座 | `outputs/rlt/base`：811 项名称/形状匹配，约 7.3 GiB |
| tokenizer | `outputs/rlt/base/tokenizer`，从已校验 SP 模型转换，抽查 token IDs 一致 |
| 精度 | 按 RLT 保留视觉、projector、norm 和动作投影 FP32，其余选定权重 BF16 |

本次短训练：VLA expert-only 3 步 loss 为 0.064、0.415、0.323，梯度有限，日志显存约
11.37 GiB；第一集提取 388 个真实 `256×2048` image-only prefix；Stage 1 原版 encoder
201,439,232 / decoder 205,631,488 参数，3 步 loss 为 1.5073、1.3983、1.0888。
这些数值仅证明链路可运行，不能据此判断收敛或任务能力。短检查 checkpoint、缓存和测试数据清理后不作为正式产物复用。

诊断日志：`outputs/rlt-environment.log`、`rlt-prepare.log`、`rlt-train-check.log`、
`rlt-prefix-check.log`、`rlt-stage1-check.log`、`rlt-stage2-check.log`。
后五个短检查日志关联已删除的临时产物，复核时重新执行短检查。

## 3. 固定观测与动作合同

| 边界 | 约定 |
| --- | --- |
| state | 从臂实测 `joint_1..joint_6,gripper`，7D |
| action | 同样 7D，绝对关节目标 rad 和夹爪宽度 m |
| camera | 唯一真实 `observation.images.cam_wrist`，256×256 RGB |
| 数据时序 | 原采样约 60 Hz；同一索引同时选 state/action/RGB，最近邻重采样 30 Hz |
| normalization | 本轮鼠标数据的 q01/q99；`use_relative_actions=false` |
| VLA chunk | 50 个 30 Hz 目标；模型输入内可补齐到 32D，物理边界仅取 7D |
| RL chunk | 10 个连续目标；当前上游 `action_stride=2` 用于 replay 窗口采样，不是跳着执行动作 |
| SDK servo | 500 Hz 内部控制，不能作为模型采样或推理频率 |

窗口内按 30 Hz 命令节拍配置；10 步的标称动作时长约 0.333 秒。
2026-10-30 当前采用窗口边界同步动作推理：VLA 参考、z_rl 与 Actor 使用同一份边界观测，
learner 更新与 rollout 采集异步。跨窗口推理等待计入实际平均频率，不用较早观测的动作预取替代边界参考。
不能直接降成 10 Hz 后仍称时间语义相同。RTC 初期关闭，先验证连续目标与执行时间对齐。

数据 action 是当时提交的目标，SDK 可能插值或限幅；在线 replay 必须记录实际接受的受限目标，
另存原始预测与测量反馈。测量值略越限与目标越限分开处理，记录浮点容差，不放宽物理限位。
不复用原双臂 14D ARX 统计或 SO-101 的角度/夹爪尺度，不复制腕图像补出另一个视角。

## 4. 环境复查与资源安排

以下命令现在可执行：

```bash
cd ../pi05
bash rlt_setup.sh
./rlt check --gpu
nvidia-smi
free -h
df -h .
```

每次长任务前复查资源；只管理本轮启动的进程，不停止其他模型、服务或用户任务。
代理优先 `http://127.0.0.1:7897`，失败后比较直连，保持错误和状态码，不改全局设置。

正式训练预计还需：全量 prefix 约 14.4 GiB，VLA 每份模型约 7.3 GiB（另有 optimizer 状态），
Stage 1 每份含 optimizer 的 checkpoint 约 4.6 GiB，best 和最终至少两份。
精确占用以首次正式保存后的 `du` 为准；另预留日志、相机录像和在线 replay 空间。
默认只保存最终 VLA，以及最终/best Stage 1。要增加周期 checkpoint 时先计算磁盘预算。
本机不能直接使用上游每 1,000 步保留一份大 checkpoint 的默认安排。

## 5. 示教审阅与正式 VLA 微调

现有数据已转换，重复运行 `./rlt data` 会拒绝覆盖。恢复产物时先检查来源与已有目录，
再选择新的输出目录转换；不删除原 HDF5。

```bash
# 初次或丢失转换产物时执行：
./rlt data
# 复用/验证本地基座，保存 7D config 和鼠标数据 processors：
./rlt prepare
```

正式训练前审阅全部 50 集：检查鼠标/鼠标垫可见、成功标签、坏帧、开始/结束区间与关节动作。
质量不合格的集应显式排除并重新生成对应数据、统计和 processors。
第一轮流程复现可用全部 50 集训练，但训练 loss 不能当独立验证指标；若做离线验证集，
必须按 episode 划分并只用训练集计算统计，不能随机切相邻帧。独立现场评估不使用训练标签冒充新结果。

先选择并记录训练变体：

1. 优先尝试原文的全参数微调：先 3 步测量峰值显存、RAM 和保存占用，检查 loss/梯度有限。
2. 若本机资源不足，保留失败日志，使用已经验证的 expert-only 跑完整流程。
3. 全参数与 expert-only 的结果分别命名和报告；不在同一个实验中悄悄切换。

```bash
# 需要首先测量的全参数短检查，当前尚未验证：
./rlt train --steps 3 --output outputs/rlt/full-vla-check

# 下面两个正式命令按所选变体只执行一个；输出目录须未被占用。
./rlt train --steps 30000 --output outputs/rlt/vla
./rlt train --expert-only --steps 30000 --output outputs/rlt/vla
```

两种变体都从本地基座开始正式训练，不把 3 步检查 checkpoint 当已完成微调。
通过条件：30,000 步完成、无非有限梯度，最终权重/optimizer 可重新加载，config 保留 7D、
单腕图像和绝对动作；保存实际训练配置与学习率轨迹。随后按第 8 节验收 VLA 真机基线。
若基线连鼠标区域都不能接近，先补数据/调整 VLA，不能依赖在线 RL 修复全部任务行为。

产物：`outputs/rlt/vla/checkpoints/last/pretrained_model` 及训练状态。
为模型和 processors 记录 SHA256，后续阶段冻结该版本。

## 6. 全量 prefix 与正式 RLT Stage 1

只使用第 5 节选定的正式 VLA，先核对同一 tokenizer、相机映射和归一化。

```bash
# 全量 50 集，默认 batch=2：
./rlt precompute
# 先作有限前后向检查，不保存训练 checkpoint：
./rlt stage1 --dry-run
# 原版 paper_v1：2048D，encoder/decoder 各 4 层、8 heads；初始 batch=1。
./rlt stage1 --steps 30000
# 正式产物配套检查，不访问硬件：
./rlt stage2-check
```

prefix 产物为 `outputs/rlt/embeddings/{prefix_out.mmap,prefix_mask.mmap,meta.json}`。
检查 `complete=true`、14,709 帧、image-only、post-transformer、有效 mask 和有限数值，
缓存中的 `pi05_path` 必须指向正式 VLA。补算或换 VLA 时使用新目录，不能混用旧缓存。
Stage 1 产物为 `outputs/rlt/stage1/best_checkpoint.pt` 和最终 step checkpoint。
batch 可用 `--batch-size` 小步增加，先测显存，默认不缩减论文网络规模。
上游 best 指训练重建 loss 最小，不能称为验证集最优或任务成功率最优。

## 7. ARX 主从运行器开发与离线验收

2026-10-21 已实现 `./rlt stage2`：`rlt_stage2.py` 连接上游 Stage 2 的 encoder、Actor、
twin critic 和 replay，`rlt_stage2_hardware.py` 独立控制 ARX 主从臂。
`rlt_stage2.json` 固定当前的 28000steps VLA 和 `stage1_pi05_28000/best_checkpoint.pt`。
VLA 词表输出头通过精确白名单兼容，运行时用到的权重仍严格加载。
SO-101 factory、Torque_Limit 和角度标定不会用于 ARX backend。

保持最小架构：RLT 模型/learner 留在 `.venv-rlt`；单独 SDK 进程使用 `arx5` Python，
独占两条 CAN；上层通过有超时的本机 RPC 获取实测状态、发送受限目标和切换主从模式。
参考 `rlt infer` 的 SDK 进程，以及 `leader_follower_teleop.py` 的主臂拖动、
米制夹爪与从臂限位处理；租约/停止采用独立于模型推理的硬件看门狗。
不重写上游 encoder、actor、twin critic 或 replay 算法。

已实现下列合同，实物表现仍需按第 8 节验收：

- controller 构造前完成两臂机型、标定、重力与限位配置；离线流程禁止构造 controller。
- follower 7D 实测反馈、时间戳、emergency 和目标回执；任何失败不能当作动作已执行。
- leader 拖动、位置保持、从臂镜像三种增益模式；不能模拟 SO 寄存器接口。
- HIL 切换从当前从臂姿态连续对齐，确认反馈后交接，拒绝不连续首个目标。
- 左/右夹爪宽度分别 0.088/0.086 m，主从宽度映射显式定义、限幅并保留原值。
- finite/shape/关节限位、按实际 dt 限速、跟踪误差、画面过期和推理超时检查。
- 推理线程不阻塞 watchdog；保持机制同时受观测/反馈时效约束，不能无限续命。
- 正常结束、异常、Ctrl+C、通信中断和租约超时的停止行为及停止确认。
- replay 写入实际接受目标和真实人工标签；自动归位关闭、回合间人工摆位。

当前配置：左全开读数 -3.2969、右 -3.45369，两臂重力倍率均为 1.0；
关节/夹爪限速为 0.15 rad/s、0.01 m/s，拖动 kp=0、kd=默认值×0.05，力反馈和速度前馈关闭。
夹爪沿用 SDK 的 1 mm 闭合阈值及米制限幅；HIL 只对六个关节保存姿态偏移，夹爪不叠加偏移。
这些是显式运行配置，现场启动前重新核对。
仅在无人驱动相同 CAN 时进行 SDK 通信验收。构造 controller 本身会收发电机，
阻尼不能证明手臂/负载被可靠支撑。

离线已用临时替身验证 7D 顺序、宽度映射、限幅回执、缺帧/过期/超时/NaN/接管/停止，
17 项检查通过；75 项相关上游回归检查通过。真实模型与 encoder 跑完 3 个模拟回合，
完成 32 次 learner 更新，检查了人工纠正和有限权重；另以 2 个模拟回合验证 Actor 执行，
22 条 `10×7` replay 和相应 checkpoint 恢复。临时测试文件、配置及模拟产物已清理。
这些证据不代表真实机器人或任务成功率验收。实际操作命令和输出见根 README。

SDK 操作依据：[硬件索引](../arx5-sdk/docs/hardware/README.md)、
[本机快照](../arx5-sdk/docs/hardware/current-machine.md)、
[动作合同](../arx5-sdk/docs/hardware/model-integration.md)。

## 8. 相机恢复、短步验收与 VLA 基线

当前 RLT 只需要 Vin 1 腕相机，不依赖原双臂 π0.5 的三路观测。
2026-10-14 Vin 1 已恢复，10 秒连续采集约 59.98 FPS，UMI 裁剪的 256×256 RGB 有效。
左右臂均通过 3 秒原地反馈检查。左臂首次开度低于 −0.005 m，触发保护；
现场确认夹爪打开后，保留原标定复测通过，开度约 87.86 mm。
该结果不证明闭合零点正确，进入夹爪运动验收前须验证闭合端。
诊断进程均已退出，未执行回零、轨迹或重新标定。
详细证据与夹爪核对顺序见 [硬件链路检查记录](HARDWARE_LINK_REPORT.md)。
每次开始真机阶段仍需重新检查相机、独占 CAN 和反馈；当前画面与示教安装的对齐尚未确认。

```bash
# 先停止独占相机和 CAN 的第三方多任务部署，再仅检查 RLT 右腕相机：
../openpi/arx5 down
cd ../arx5-sdk/python
${ARX_PYTHON:-python} examples/record_episodes.py --check-cameras --no-preview
cd ../../pi05
```

这能验证采集卡输出；RLT 运行器还必须验证与示教相同的 RGB、UMI resize+中心裁剪到 256、
物理安装、帧率和新鲜度。仅相机打开成功不算通过。

按顺序完成唯一 CAN writer 检查、真实反馈/有效配置查询、当前位置保持、逐轴极小运动、夹爪开闭、
主从拖动、mirror、接管与停止。每一步同时保存目标和实测反馈；发送 OK 不等于到位。
现场检查通过后先 `vla_only` 跑短窗口，再跑完整鼠标任务。固定测试起点采集至少 20 次基线回合，
记录成功/失败、介入次数、耗时、轨迹、相机和停止结果。
在线 RL 前需证明 VLA 可到达有效操作区域。动作质量不足时返回第 5 节。

## 9. 手动关键阶段与在线 RL

第一轮用 `c` 人工切换，完成核心 RL 流程。鼠标任务的候选关键阶段是最终对准、夹取、
落到鼠标垫并释放；先根据演示确定实际边界，不凭文本自动生成奖励或区间。
边界索引采用转换后 30 FPS 的 episode-local 帧号，不能直接使用原 60 FPS 索引。

本机配置为 `rlt_stage2.json`，运行参数如下：

| 参数 | 初始安排 |
| --- | --- |
| VLA/RLT/tokenizer/stats | 第 5、6 节同一版本产物，7D 状态/动作 |
| task/camera | 鼠标指令，cam_wrist → observation.images.cam_wrist |
| control_fps | 30，实测确认时序后固定 |
| RL chunk / replay stride | 10 个连续动作 / 2，保持上游机制 |
| mirror/HIL | ARX 增益与姿态对齐接口，第 7、8 节验收后启用 |
| reset/manual positioning | 自动 reset 关闭，manual positioning 开启 |
| RTC | 初期关闭，后续有单独时序证据再启用 |
| warmup | 至少 20 个纯 VLA 回合，关键阶段激活并保存成败 |
| learner | 原版 full-output actor、twin critic、reference dropout=0.5、UTD=5、critic/actor=2 |
| 运行长度 | 先 2–5 个链路回合；再固定约 200 个正式回合并记录实际样本量 |

先 dry-run，再 VLA warmup，检查 replay 内目标、奖励、terminal boundary、人工介入和 valid mask。
达到配置的 replay/learner 更新条件后，`--auto-critical` 模式在关键阶段自动允许 Actor 接管，无需按 `a`；
`a` 保留为禁止/重新允许开关。逐批观察 Q、actor/critic loss、BC 偏离、
动作限幅与真实成功率；不因 loss 下降就报告任务改善。
真实 y/n 结果写入奖励，丢弃回合和通信故障保留原因。
中途 checkpoint 保留运行配置和 replay 索引；续训用新输出目录，显式恢复 checkpoint 和 replay。
正式产物为 `outputs/rlt/stage2/final_checkpoint.pt` 及 replay、日志、版本清单。

## 10. 自动关键阶段分支

完成手动链路后，补齐可选分类器以完整运行上游流程。
先适配上游标注工具读取当前 image dataset（原工具按 video metadata 找文件），
以及 `precompute_critical_zrl.py` 的单腕相机映射和硬编码 6D proprio。
分类器训练/评估按 episode 留出至少 6 集，避免邻帧泄漏；不伪造关键区间。

适配完成后的操作顺序：

```bash
source .venv-rlt/bin/activate
python .deps/rlt/scripts/annotate_dataset.py \
  --dataset_path outputs/rlt/dataset --camera observation.images.cam_wrist \
  --annotations_out outputs/rlt/annotations_mouse.json \
  --sheet_dir outputs/rlt/annotation_sheets
python .deps/rlt/scripts/precompute_critical_zrl.py \
  --pi05_path outputs/rlt/vla/checkpoints/last/pretrained_model \
  --tokenizer_path outputs/rlt/base/tokenizer \
  --rlt_checkpoint outputs/rlt/stage1/best_checkpoint.pt \
  --dataset_path outputs/rlt/dataset --annotations outputs/rlt/annotations_mouse.json \
  --output outputs/rlt/critical_zrl_mouse.npz --batch_size 2 --device cuda
python .deps/rlt/scripts/train_critical_classifier.py \
  --precomputed_zrl outputs/rlt/critical_zrl_mouse.npz \
  --output outputs/rlt/critical_classifier_mouse.pt --val_episodes 6 --device cuda
python .deps/rlt/scripts/evaluate_critical_classifier.py \
  --precomputed_zrl outputs/rlt/critical_zrl_mouse.npz \
  --classifier outputs/rlt/critical_classifier_mouse.pt --val_episodes 6 --device cuda
```

通过条件：输出 `z_rl(N,2048)`、`proprio(N,7)`；保存留出集 precision/recall、误触发、漏触发、
起止延迟和选定阈值。先在 shadow 中核对，再在同一在线/部署入口启用，保留手动 c 覆盖。
不采用默认阈值就直接让分类器接管所有物理动作。

## 11. 导出、部署与独立评估

将 inference、export 和 verify 中 SO-101 名称与 6D 默认恢复逻辑一并适配 ARX7，
同样走第 7 节 SDK 边界，不复制第二套控制代码。
只有 Stage 1 基座或随机 actor 不构成训练好的 RLT 策略，部署必须提供正式 Stage 2 checkpoint。

完成适配后执行：

```bash
source .venv-rlt/bin/activate
python .deps/rlt/scripts/export_rlt_pretrained.py \
  --stage1-checkpoint outputs/rlt/stage1/best_checkpoint.pt \
  --stage2-checkpoint outputs/rlt/stage2/final_checkpoint.pt \
  --pi05-path outputs/rlt/vla/checkpoints/last/pretrained_model \
  --output-dir outputs/rlt/export --device cpu
python .deps/rlt/scripts/verify_rlt_checkpoints.py \
  --pretrained-dir outputs/rlt/export/pretrained_model --device cpu
```

导出不包含冻结 VLA，要随部署保留它、tokenizer、processors、Stage 1/2 来源与 SHA256。
离线 strict reload 后重启进程，在不学习的 inference 模式运行。
以相同起始位置分布分别评估 VLA-only、RLT 手动关键阶段、RLT 自动关键阶段，每组至少 20 个新回合；
报告成功率分母、人工介入率、任务耗时、限制/停止触发次数和全部失败案例。
样本少时注明不确定性，不将训练回合成功率作为独立评估结果。

最终交付：README 可独立使用的启动/停止命令、正式 checkpoint 与来源清单、
主从有效配置、相机变换、replay/评估结果和剩余问题；清理本轮测试目录，保留真实采集和诊断日志。
各阶段通过后小步提交英文一行 commit；若配置远端且有新提交，先 rebase 保持线性历史。
