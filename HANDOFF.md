# ARX5 RLT 接手上下文（2026-10-30）

## 2026-10-31 在线 RL 检查

检查 `stage2_arx_20261031_230555/checkpoint_ep105.pt`：learner 和 Actor 下发均实际运行，
23804 条 replay 含 7124 个 HIL 窗口；最近 20 回合无 HIL 仅 3/13 成功。
发现 partial terminal chunk 的 BC 屏蔽 padding，但 Actor Q 未屏蔽，导致未执行输出失控；
同权重对照最高 Q 从 16896.8 降到 1.053。现已将 Actor Q 输入按 valid_mask 置零，
51 项原有单测与真实 replay/网络的 CPU 更新检查通过。未停止现场进程或启动真机评估。
历史权重曾实际请求 -449 rad 关节目标，硬件限速拦截；旧权重不会随损失修复自动恢复。
另有过滤前 Actor 请求与过滤后 replay 动作不一致、异步 checkpoint 序列化交错，尚未修改。
证据和后续评估边界见 [在线检查报告](outputs/rlt/online_audit_20261031/README.md)。

## 目录职责

用户明确要求：RLT 统一在 `<repo-root>`；
`../pi05` 仅用于 π0.5 基座、SFT 和纯 VLA 推理。
第三方双臂模型部署在 `../openpi`，本轮未修改。

RLT 的 ARX 适配已直接并入本仓库的 `scripts/train_rlt_stage2_pi05.py`、
`scripts/rlt_core.py` 和 `lerobot/src/lerobot/policies/pi05/modeling_pi05.py`。
本机入口是根目录 `rlt` / `rlt.py`，独立 Python 环境为 `.venv`。
`pi05` 的入口已分离为 `./pi05 infer` 等，不再有 `./rlt`。

## 当前资产

| 资产 | 当前路径 |
| --- | --- |
| 冻结 VLA | `../pi05/checkpoints/28000steps/pretrained_model` |
| 50 条、14,709 帧、30 Hz 示范 | `../pi05/outputs/pi05/dataset` |
| tokenizer / π0.5 基座 | `../pi05/outputs/pi05/base` |
| prefix 缓存 | `outputs/rlt/embeddings_pi05_28000` |
| Stage 1（30,000 步，已完成） | `outputs/rlt/stage1_pi05_28000/best_checkpoint.pt` |
| 默认关键阶段分类器 | `outputs/rlt/critical_mouse_20261022_181349/critical_classifier.pt` |
| 分类器改进实验 | `outputs/rlt/critical_review_20261029_204952/` |
| 时延测量 | `outputs/rlt/timing_20261029_212023/` |
| Stage 2 配置 | `rlt_stage2.json` |
| 原会话及迁移记录 | `outputs/rlt/stage2_arx_*`、`outputs/migration_20261029/` |

35 项 RLT 产物同盘移动，没有复制大型 checkpoint。
`outputs/migration_20261029/path_mapping.json` 记录历史路径映射。
原训练日志、run_config、checkpoint/NPZ 内部的历史来源不改写；当前启动以配置和 README 为准。
π0.5 原单臂会话在 `../pi05/outputs/pi05/inference`；用户未跟踪的杯碟数据保持原样。

## 分类器与训练现状

- 50 条示范有连续人工关键区间，第 6–49 集训练，第 0–5 集验证。
- 冻结 VLA 和 Stage 1；`2048D z_rl + 7D 原始状态` 输入 `2055→256→2` MLP。
- Adam lr=1e-3，交叉熵 + 0.05 时间变化惩罚，60 轮，保存最佳第 10 轮。
- 原始验证准确率 95.56%；每 10 帧回放原检测器 F1=89.66%、召回率=85.89%、平均进入晚 0.606 秒。
- 三折交叉验证仅使用原 44 条训练回合。保留原权重，候选检测参数 `.65/.45/smooth=1`：
  原 6 条验证回合 F1=93.16%、召回率=92.70%、平均进入晚 0.272 秒，误报 46→51 帧。
- 正则化重训选出的新模型准确率 94.36%，没有替换原权重。
- 默认检测参数仍 `.75/.45/smooth=4/min_on=0`。候选配置在改进实验目录。
- 原 6 条验证回合曾用于模型选择，仍需新回合独立测试；上述结果不是任务成功率。
- 已修复在线输入错误：检测器接收原始关节弧度/夹爪米数，由其保存的训练均值/标准差归一化。
  Actor/Critic 的分位数归一化保留；分类器在连接硬件前加载和检查维度，dry-run 也覆盖。
- 历史完整真机会话未产生有效 RL 更新；最近完整日志因主臂夹爪负开度反馈锁故障退出。

## 启动与操作

```bash
cd "$(git rev-parse --show-toplevel)"
./rlt check --gpu
./rlt stage2-check --auto-critical
./rlt stage2 --auto-critical --episodes 5
```

启动前释放其他 CAN/相机持有者，现场核对接线、标定与停止行为。完整操作见
[ARX5_README.md](ARX5_README.md)。当前默认配置为单臂 `X5_umi/can0`；历史双臂配置是
`X5_umi/can1` 从臂、`X5/can0` 主臂。单臂模式下策略与 HIL 共用 can0，关节安全限速 3.6 rad/s、
夹爪 0.01 m/s；既有历史会话可能采用不同 CAN，不能把历史配置当作当前现场接线确认。

交互终端中的命令单键即时生效、无需回车：`p/p` 摆位并启动；`h/s` 人工接管/交还；训练时 `c` 手动开启/截断关键阶段，`f` 结束回合并评价；`a` 禁止/重新允许 Actor；
`f` 结束，`y/n/d` 成功/失败/丢弃；`q` 正常退出保存。Stage 2 的 `s` 不是单臂暂停保存。
Stage 2 终端按 `q` 正常退出会缓慢回到启动时记录的 home pose，再进入阻尼；急停或硬件故障不会追加回位动作。
前 20 回合采集，replay≥256 后训练。
`--auto-critical` 启动时默认允许 Actor；≥1000 更新后仍需通过 batch 均值
`Q(actor)-Q(reference)>actor_gate_advantage` 才接管，否则保持 VLA，无需按 `a`。
分类器只触发进入关键训练阶段，不因概率下降退出；到达 900 步仍继续，直到手动 `c` 关闭或 `f` 结束。
`c` 在动作窗口边界截断、锚定终点并交回 VLA，自动检测不重新开启；仍由回合末 `y/n` 提供奖励。
每窗口用同一观测检测一次，每回合重置检测历史。reference dropout=0.5、在线 BC 权重=10，
执行时保留 reference。现有 checkpoint 可继续恢复，旧 replay 保留原边界，新回合使用新规则。
`a` 手动禁止会跨回合保持，需再次按 `a` 才重新允许；手动关键阶段模式仍从禁止状态启动。
本轮 51 项原有单测、16 项自动许可/执行门控/单键覆盖检查及根入口只读预检通过，
证据保存在 `outputs/rlt/auto_actor_20261030/`；没有连接 CAN 或实时相机，没有新增永久测试文件。
当前配置使用 RTC guided horizon=10、30 Hz 直发和无 EMA chunk filter；探索噪声按 chunk 广播，
Actor 输出钳位 ±1.5，BC/Q 权重从 10/1 退火到 2.5/4。正式训练为
`./rlt stage2 --auto-critical --episodes 200`，恢复用 `--resume-from`，
`--episodes` 是回合总编号上限。没有启动本轮真机运动。

## 检测与推理频率结论

2026-11-01 当前执行方式：已恢复 `async_inference=false`、`replan_every_window=true`，RTC 使用 guided horizon=10。
每个 10 步窗口结束后，用同一份边界图像和实测状态同步生成 VLA 参考、z_rl 与 Actor 动作，
再下发下一窗口的 10 步；推理期间维持最后接受的目标，不提前推进旧动作块。
learner 的独立更新线程保留，rollout 与学习仍异步。C=10、replay stride=2、RTC guided 及单键操作保留。
窗口内按 30 Hz 提交目标，跨窗口的推理等待计入实际平均频率。本轮 51 项原有单测和根入口只读预检通过，
未连接 CAN 或实时相机。旧异步会话 checkpoint 的 provenance 与同步配置不同，仍须通过恢复兼容性检查。

以下为 2026-10-29 动作预取试验的历史记录，当前默认已关闭；原因是提前观测生成的参考与 Actor 边界状态不同时。
试验当时使用 `async_inference=true`，保持 C=10、replay stride=2、RTC 关闭。
下一份 50 步 VLA 参考按上一轮耗时提前生成，窗口满 10 步才换块。当前动作和中间 replay 快照放在 CPU，
避免每步复制 CUDA 张量时等待后台推理。模型调用串行化，实时推理优先于 replay 特征物化；learner 在
预取期间让出新更新机会。实际窗口边界仍从新观测提取 z_rl，VLA 参考的预取延迟记录在 planning.jsonl。
首块/迟到时保持并继续处理接管及退出，等待上限 2 秒；接管、回合尾与退出均使旧计划失效。
这项适配保留论文的训练窗口，并引入参考观测延迟，不代表完全复刻其部署时序或保证硬实时。
同轮验收发现固定输入的 SigLIP SDPA 串行输出会波动；仅将 Stage 2 视觉塔切为 eager 后，完整 z_rl 的
重复对照最大差值为 0。权重、精度、分类器及检测参数未改。执行模式与视觉后端写入 runtime provenance，
恢复旧 Stage 2 checkpoint 时仍须通过兼容性检查，不能混用不一致的运行来源。
51 项原有单测、30 项临时异步检查、4 个真实权重/保存帧/模拟动作下发窗口及根入口只读预检均通过。
后台 VLA 推理约 89 ms，新鲜边界特征约 33 ms；证据位于 outputs/rlt/async_window_20261029/，临时测试脚本
验收后删除，既有上游 tests 保留。这轮未接入实时相机、CAN 或在线 learner，现场频率尚未测量。
当前入口及配置见 ARX5_README.md；下面的耗时表和同步会话分析是历史测量。

对应用户关于每 10 帧检测、逐帧开销、精度与 30 Hz 的问题。
代码在每 10 个动作窗口的边界检测；部分边界前后两次调用，前一次复用上个 z_rl，
后一次使用新 z_rl。这不等同于稳定每 10 个相机帧检查一次，实际受同步推理停顿影响。

在本机 5090、28,000 步模型、保存的真实 256×256 RGB 单腕输入上，RTC/compile 关闭，
CUDA 同步、3 次预热后 15 次测量中位数：

| 工作 | 耗时 |
| --- | ---: |
| π0.5 完整 10 步去噪（已处理输入） | 85.09 ms |
| 单臂预处理 + π0.5 + 后处理 | 86.41 ms |
| Stage 2 完整计划 + 新 z_rl | 119.94 ms |
| 跳过去噪，仅刷新图像特征 + z_rl | 33.16 ms |
| 已有 z_rl 的分类器 update（300 次） | 0.054 ms |

参数实测约 2.820B 为 BF16、0.534B 为 FP32；没有 INT8/INT4。
50 步 chunk 的动作时间轴为 30 Hz；不是完整模型每秒运行 30 次。
每帧 MLP 本身便宜，但每帧刷新 z_rl 近乎占满 33.3 ms 的预算，还要与 VLA/learner 共用 GPU。
复用旧 z_rl 每帧检查状态，不能称为每帧重新理解新图像。

历史 20,000 步纯 VLA 真机会话 247 次预取：中位 111.26 ms，P95=115.41 ms，首推理353.11 ms。
其请求 100 Hz 的命令，按命令回执 SDK 反馈时间戳计算实际间隔中位约 36.89 ms（27.1 Hz）；
当前单臂循环在 RPC/预览后还 sleep 整个周期，不能把 `--control-hz` 当作实测频率。
Stage 2 的实际时钟频率尚未在本轮硬件上测量。不要把 `30/10=3 Hz` 当作实测检测频率。

## 验证

迁移后的两边环境检查、CUDA BF16 运算、RLT 真实 VLA/encoder/分类器 dry-run 已通过，
日志在 `outputs/migration_20261029/`。51 项现有 Stage 2 单元测试通过，迁移后的保存图像完整推理基准也已通过。
临时测试用临时目录或内存执行，不新增永久测试文件，不删除已有上游 tracked tests。
