> 2026-10-29 随 RLT 迁入本仓库。本文是历史硬件验收证据；当前目录、标定与启动入口见 [ARX5_README.md](ARX5_README.md)。

# ARX 主从臂硬件链路检查

检查时间：2026-10-14 14:06–14:14；左臂复测至 14:25，主从测试至 14:51（Asia/Shanghai）。
2026-10-14 结论：Vin 1 采集、左右臂初始化、原地实时反馈及基础人工主从遥操均通过。
左臂首次因夹爪位置保护失败；现场确认夹爪打开后，保留原标定复测通过，闭合零点仍待验证。
人工拖动跟随、暂停/恢复及正常退出已有实测；RLT 策略接管、超时保护与鼠标任务闭环尚未验收。

## 2026-10-22 CAN 角色修正

用户现场确认 `can0` 是主臂、`can1` 是从臂。此前 Stage 2 配置反向，已修正为
主臂 `X5/can0`、从臂 `X5_umi/can1`，单臂 VLA 的默认接口也改为从臂 `can1`。
既有主/从臂机型和标定随角色应用；同一个 `can1` 原始角度 `-3.499084 rad`
按从臂 `-3.45369 rad / 0.086 m` 换算约为 `0.087130 m`，处于 SDK 范围容差内。
此前操作者开合主臂时一直查看 `can1`，不能据此认定主臂反馈冻结。
下文 2026-10-14 和 2026-10-21 的 CAN/角色描述是当时使用的配置，当前启动以新配置为准。

本次原生 SDK 配置核验及 28000steps VLA / Stage 1 encoder 的 CPU dry-run 通过。
只读查询两路各 30 帧：主臂 `can0` 约 0.005 mm，从臂 `can1` 约 78.6 mm，均在 SDK 容差范围内；
证据为 `outputs/rlt/gripper_check_20261022_175602.jsonl`。随后操作者启动的
`outputs/rlt/stage2_arx_20261022_175615/` 日志确认两只控制器初始化成功，
之后因启动相机帧龄 `0.308s` 超时退出，尚未进入在线训练回合。

## 2026-10-21 Stage 2 启动阻塞

当时被配置为主臂的 `can1` 多次报 `Gripper position error: got 0.093 but should be in 0~0.088`，尚未进入回合。
本次绕过 SDK controller，仅向 `0x7FF` 发送 `0800cc0000000000` 状态查询。
`can1` 的 8 号电机实时响应角度为 `-3.499084 rad`，按当前 `-3.2969 rad / 0.088 m`
标定换算为 `0.093396655 m`，与 SDK 报错一致；`can0` 同时返回接近零的独立角度。
两路各 750 个响应、75 秒内，`can1` 角度范围为 `[-3.499466, -3.499084] rad`。
尚未收到操作者对本次采样期间实物开合的确认，不能单凭静止读数认定编码器损坏。

证据在 [`outputs/rlt/gripper_feedback_20261021_221207/`](outputs/rlt/gripper_feedback_20261021_221207/)：
`raw_feedback.json`、`manual_feedback.json` 和 `cli_feedback.jsonl`。新增 `./rlt gripper-check --seconds 60`
可在开合两只夹爪时核对实际 CAN 对应关系，操作见 README；17 项离线检查及真实查询通过，
退出后无残留 CAN 接收 socket。本次没有构造控制器、使能、回零、重设零点或改动标定。
Stage 2 真机控制仍未验收。

## 2026-10-14 检查范围

按现场确认执行 SDK 初始化和原地反馈检查。左 X5/can1 作主臂，右 X5_umi/can0 作从臂。
两臂逐个检查，先右后左；构造前配置关节 kp=0、默认 kd、关闭重力补偿、退出转阻尼。
右臂实际关节 kd 为 `[2,2,2,1,1,0.7]`，夹爪 kp/kd 均为 0。
没有调用回零、轨迹、关节位置命令或重新标定。SDK 初始化仍会清电机状态、使能并收发电机。

## 本次结果

| 链路 | 实测结果 | 验收范围 |
| --- | --- | --- |
| CAN 接口 | can0/can1 均 UP、LOWER_UP、ERROR-ACTIVE；错误、丢包、bus-off 均为 0 | 接口和本次通信通过 |
| 右 X5_umi/can0 | 3 秒、297 个状态样本；六关节和夹爪均有重复 CAN 回复，未触发 emergency | 原地反馈通过 |
| 左 X5/can1 | 首次夹爪保护阻止初始化；打开状态下复测 3 秒、291 个状态样本，七电机持续回复，无 emergency | 原地反馈通过，闭合零点待验证 |
| Vin 1 腕相机 | 10 秒约 59.98 FPS；256×256 RGB，UMI 等比 resize/中心裁剪 | 采集通过，任务安装对齐待核对 |
| 原策略服务 | 原 PID 3390704、端口 8005，`/healthz` 曾返回 `OK` | 2026-10-15已停止，第三方多任务部署迁至openpi |

右臂实测夹爪宽度约 0.07165 m，在 0–0.086 m 内；六关节位置均在配置范围内。
最大关节变化量 0.0003815 rad，最大 SDK 状态快照 age 为 3.778 ms。
SDK 时间戳为软件更新时间，不能单独证明各电机反馈新鲜；本轮同时检查原始 CAN 回复。
完整初始化、采样和退出窗口内，EC IDs 1/2/4 分别收到 1484/1484/1477 帧；
DM IDs 5/6/7/8 分别有 1494/1493/1492/1492 帧状态回复。
这些计数包括退出阻尼阶段，不能直接作为 3 秒采样的电机频率。

相机接收后到读取时的 host age p50/p95/max 为 10.86/18.36/21.93 ms，
没有超过 30 ms 的轮询；该值不是曝光到主机的总延迟。
首末图已目视确认能看到工作台、鼠标垫和夹爪，不是历史均匀黑帧。
本轮没有确认鼠标任务摆位、相机与示教安装一致，也没有验收 Vin 4 或手机主相机。

## 左夹爪首次保护记录

本轮使用历史参数：左 `gripper_width=0.088` m、`gripper_open_readout=-3.2969` rad；
右分别为 0.086 m、−3.45369 rad。左初始化原始错误：

```text
Gripper position error: got -0.005 but should be in 0~0.088 (m).
Emergency latched: damping only; motion rejected until controller restart.
RuntimeError: Emergency latched; restart the controller after resolving the fault
```

日志开度只保留三位小数；源码保护条件是开度严格小于 −0.005 m，或大于全开宽度加 0.005 m。
换算为 `motor_angle / gripper_open_readout * gripper_width`，负开度不能解释为实物的负宽度。
首次失败时零点偏移、实物位置与标定是否匹配尚未确定，当时没有重试初始化或修改参数。

后续现场反馈夹爪已打开，复测结果见下节。若需要验证闭合零点或重新标定，按
[SDK 标定流程](../arx5-sdk/docs/hardware/setup.md#calibration) 单独安排现场操作；
初始化失败时不直接反复运行标定脚本。进入运动验收前，仍需核对闭合零点和全开实物宽度。

## 左臂打开状态复测

现场确认夹爪打开后，使用同一 `-3.2969` rad 标定、0.088 m 宽度及阻尼配置复测。
没有修改 SDK、标定、保护阈值，没有回零或发送位置/轨迹目标。
有效检查退出码 0：3 秒、291 个状态样本，关节及夹爪反馈有限且位于配置范围内，
未触发 emergency。采样阶段电机 IDs 1/2/4/5/6/7/8 各收到 1463 帧回复。
夹爪换算开度为 87.858–87.868 mm，换算电机角度约 −3.29156 rad，接近历史全开读数；
这支持打开端的读数一致性，不能证明闭合端零点正确。
最大关节变化量 0.0003815 rad，最大 SDK 状态快照 age 为 2.122 ms。

第一轮复测已成功初始化，但诊断的 Python CAN 观察线程尚未处理首批帧，导致提前退出；
其汇总又因空集合报错，日志保留在 `outputs/hardware-left-recheck-20261014_141935/`。
该次不能当作硬件反馈失败或通过。有效检查改用同步 CAN 读取，消除线程启动时序影响，
不将读取缓冲区的时间解释为电机回复的硬件时间戳。
右臂和相机本次未重复检查，继续引用本日首轮结果。

有效复测证据：[outputs/hardware-left-recheck-20261014_142256](outputs/hardware-left-recheck-20261014_142256/)，
含 `motor-feedback.log`、`left-feedback.json`、`exit-code.txt` 和 `post-check.json`。
退出日志确认阻尼及后台线程 join；随后两条 CAN 5 秒无帧、接口计数无变化、无残留接收 socket，
8764/8765 无监听，原 PID 3390704 的策略服务仍返回 `OK`。两次复测均已退出，无永久测试文件。

## 主从遥操预检

2026-10-14 检查 `../arx5-sdk/python/examples/leader_follower_teleop.py`：
独立 CLI 的 `--help`、桌面键盘依赖可用，CAN 无其他控制器占用，原策略服务保持健康。
源码及替身复现确认原 q 退出后控制线程仍存活；已修复为停止并 join 控制线程，
正常退出再尝试回零，finally 请求阻尼并显式释放两臂控制器。故障路径不额外回零。
本次没有新增控制入口，也没有改变主从标定或 recorder 生命周期。
SDK 修复提交为 `c997e52`，保留在本机，未推送远端。

8 组离线检查通过：q、键盘循环 Ctrl+C、键盘异常、控制异常、启动回零异常、退出回零异常、
阻尼异常，以及关节映射/限位/速度前馈/夹爪裁剪/主臂增益/力反馈限幅/暂停/重复关闭。
这些检查使用替身，不连接 CAN，不证明真机到位。测试以内联命令执行，未保留测试源码。
日志在 [outputs/teleop-check-20261014_143436](outputs/teleop-check-20261014_143436/)。

预检时真机主从回零和跟随尚未执行：此前原地反馈现场确认未覆盖回零及跟随路径。
按 [SDK 操作流程](../arx5-sdk/docs/hardware/operations.md#preflight)，
现场需确认两臂整段路径空闲、有操作者值守急停并可轻微拖动主臂。
确认后先检查回零实测到位与暂停保持，再作小幅跟随、暂停/恢复和正常退出。
首次跟随用 `--vel-ratio 0 --force-scale 0` 排除速度前馈与夹爪力反馈影响；
验证后再分别启用。夹爪闭合零点仍待验证，本轮不以抓取物体作为验收方式。

## 主从真机与人工测试

现场授权运动且有操作者值守后，执行下列检查；全部进程均已退出，未重启原策略服务。

1. 首次限速运行在启动回零后未达到反馈门槛，未执行跟随即退出。原 SDK 回零函数返回后，主臂仍未进入要求的位置范围；新增反馈等待后再切换拖动模式。
2. 第二次自动分段运行完成启动、两段跟随、暂停/恢复及退出：12,493 次跟随命令，1,778 个采样快照，无控制故障。提交目标与主臂原生关节角一致，夹爪裁剪至从臂 0.086 m。两段最大关节跟随误差分别约 0.01984 / 0.01259 rad。
3. 第二段主臂第二关节变化约 0.057984 rad，从臂约 0.061799 rad。操作者明确确认没有拖动，是主臂自行移动；当时主重力倍率为 1.2。这只能证明从臂跟随输入，不能算人工遥操通过。
4. 改用主重力倍率 1.0 作 6 秒局部观察，299 个样本，各关节位置范围最大约 0.009537 rad；它不证明全部姿态或长时稳定性。
5. 随后在桌面终端启动真实 CLI，操作者手动测试并反馈“没有问题”。日志记录 space 开始、暂停、关节跟随及正常退出。启动和退出回零均通过新位置反馈门槛，未记录 emergency 或控制异常；两臂阻尼、后台线程 join、CAN socket 销毁后打印 `Done.`。本轮基础人工主从遥操按现场反馈通过。

人工测试有效命令（在 SDK 的 `python` 目录，使用 arx5 环境）：

```bash
python -u -B examples/leader_follower_teleop.py X5 can1 X5_umi can0 \
  -g -3.2969 --follower-gripper-open-readout -3.45369 \
  --leader-gravity-scale 1.0 --follower-gravity-scale 1.0 \
  --joint-speed-limit 0.15 --gripper-speed-limit 0.01 \
  --vel-ratio 0 --force-scale 0 --diagnostics
```

两臂标定未修改。速度参数是 SDK 目标速度上限，不是自由拖动主臂实测速度上限。
回零等待检查关节绝对位置 <=0.03 rad、夹爪距全开 <=3 mm、有限值、SDK 状态更新时间与 emergency；
该门槛不是精确零位或速度静止验收。更新仅影响独立 teleop，不改变 recorder 生命周期或默认重力倍率。
SDK 提交 `8de7590`（本机保留，未推送），已通过反馈等待、速度参数和 CLI 退出离线回归；没有保留测试源码。

人工测试后独立被动观察：can0/can1 连续 5 秒无帧、接口计数不变、无残留 CAN 接收 socket，
主从进程已退出；原 PID 3390704 的 `/healthz` 返回 HTTP 200 / `OK`。
本次不验收夹爪闭合零点、夹持物体、夹爪力反馈、速度前馈、所有姿态稳定性或长时运行，
也不表示 ARX RLT 在线运行器与策略人工接管已经完成。

本机证据：

- 首次启动失败：[outputs/teleop-live-20261014_143953](outputs/teleop-live-20261014_143953/)。
- 自动分段跟随与自行漂移：[outputs/teleop-live-20261014_144348](outputs/teleop-live-20261014_144348/)。
- 倍率 1.0 局部观察：[outputs/leader-stability-20261014_144812](outputs/leader-stability-20261014_144812/)。
- 人工测试及退出核对：[outputs/teleop-manual-20261014_144947](outputs/teleop-manual-20261014_144947/)，含 `launch.json`、`teleop.log`、`cli-offline-regression.json`、`stop-request.json`、`post-check.json`。操作者结束后进程已退出，未再发送 SIGINT。

## 退出与证据

相机检查进程退出码为 0；电机检查因左臂初始化失败退出码为 1。
右臂退出日志确认转阻尼并 join 后台线程；左臂构造失败后 CAN receiver 和 socket 已销毁。
原始 JSON 中 `local_tx_frames_after_release_500ms` 含观察线程延迟处理的可能，
不能凭该字段断言退出后继续发送。独立退出检查确认无残留 CAN 接收 socket，
5 秒被动观察两条总线均无帧，接口计数也没有变化；8764/8765 无监听。
电机最后的阻尼状态不代表主动位置保持或断电。

本机证据目录：[outputs/hardware-link-20261014_140659](outputs/hardware-link-20261014_140659/)。
关键文件为 `motor-feedback.log`、`right-feedback.json`、`left-feedback.json`、
`vin1-stream.json`、`vin1-first.png`、`vin1-last.png`、`can-status.json` 和 `post-check.json`。
证据受 `outputs/` 忽略规则管理，保留在本机；本轮未创建永久测试脚本或测试目录。

## 2026-10-19 单臂 VLA 短窗口

使用 `../pi05/checkpoints/5000steps/pretrained_model` 做了 30 步
VLA-only 单臂运行，任务指令为 `put the mouse on the mouse pad`。右臂 `X5_umi/can0`
和 Vin 1 均由新的 RLT worker 独占；相机实际为 `1920×1080 MJPG/60 Hz`，UMI
resize-preserve-aspect + center-crop 后输出 `256×256 RGB`，首个反馈 camera age 为
17.6 ms，机器人反馈 age 为 2.0 ms，emergency=false。

本轮策略输出和实际反馈均有限，30 条目标全部提交；关节实际最大逐步反馈变化约
`[0.0515, 0.0645, 0.0381, 0.0530, 0.0408, 0.0446] rad`，夹爪从约 84.5 mm 到
85.6 mm。提交目标与反馈的最大误差分别低于 `0.139 rad` 和 `3.7 mm`，没有触发
限位、跟踪误差、相机过期或 emergency。达到 30 步后进入 hold，退出时 worker 转阻尼，
CAN 接收线程正常 join；这证明软件到真机的短链路和保护逻辑，不能作为鼠标任务成功率。

证据：[../pi05/outputs/pi05/inference/20261019_125400_271248](../pi05/outputs/pi05/inference/20261019_125400_271248)，
包括 `actions.jsonl`、`events.jsonl`、`hardware_info.json` 和 `hardware_worker.log`。
