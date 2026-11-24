# 智训引擎 · B 题源代码包

赛题：B 题 轴承健康评估与剩余寿命预测（RUL，单位分钟）
队伍：北京工商大学 · 智训引擎队
版本：R5 / 2026-11-24（镜像 0.5）

## 一、这个包里有什么

  baselines/        官方基线代码（其中 model.py 已按约定加入输入整形与目标变换）
  harness/          无人干预闭环：观察 -> 提案 -> 执行 -> 评估 -> 门控 -> 记录
  docs/             设计说明、P1/P2 实验记录、环境差异核查、模型配置
  evidence/         容器内实跑留档的决策日志（不含任何凭证）
  selfcheck.py      容器自检（10 项，合成数据，无外部依赖）
  pytrio_probe.py   大模型连通性探测（四步递进，纯标准库）
  mock_pytrio.py    本地模拟大模型服务，无凭证时验证链路
  Dockerfile        镜像构建脚本（linux/amd64，Alpine 瘦身版，导出 40.5 MB）
  Dockerfile.slim   同内容但用 Debian slim 基础镜像（导出 64.7 MB，仅备用）
  build_image.sh    构建 + 自检 + docker save 一条龙
  to_classic_tar.py Docker 29 的 OCI 布局转经典 v2 布局
  checkpoint.npz    赛题下发的起点检查点（未替换）
  training.json     官方基线指标（RMSE 681.16 / MAE 563.11）

## 二、不包含什么（以及为什么）

1. 公开训练/验证集（public/）：属赛题方提供的数据，不在本包内重复分发；运行时由镜像或平台提供。
2. 运行产物（runs*）：只保留 evidence/ 下三份决策日志作为过程证据。
3. 任何 API Key / .env 文件：凭证一律运行期注入，不入包、不入镜像。
4. 探索期的实验台脚本（_exp/）：一次性脚本，不参与最终方案，未纳入交付。

## 三、三条复现命令

  # 1) 容器自检（10 项，不需要数据）
  python selfcheck.py

  # 2) 本地跑完整闭环（需要 public/train.jsonl 与 checkpoint.npz）
  python -m harness.loop --data public --ckpt checkpoint.npz --run-dir runs_demo

  # 3) 只看不跑（观察 + 提案）
  python -m harness.loop --data public --ckpt checkpoint.npz --dry-run

依赖只有一个：numpy==1.26.4（见 baselines/requirements.txt）。Python 3.12。

## 四、本方案的四个改动在哪

1. 通道内归一（abs8）：baselines/model.py，ZXJ_INPUT_MODE=abs8 / abs8log
   对每通道 8 个绝对量做逐样本标准化，消掉跨轴承尺度差。
2. 14 列对数绝对量（logabs14）：baselines/model.py，随 ZXJ_INPUT_MODE=abs8log 启用
   把被归一掉的能量水平以相对形式补回；外部输入仍是官方 34 维，派生列由代码现算。
3. 目标幂（y'=y^p，默认 p=0.45）：baselines/model.py，ZXJ_TARGET_POWER
4. 多轨迹权重平均：baselines/train.py，ENSEMBLE_BATCH=96,112,128,144,192,256
   同一续训起点跑 6 条轨迹，训练后对权重与 Adam 状态取算术平均。

全部开关都可关闭以复现官方或中间版本行为：

  # 官方基线行为
  ZXJ_INPUT_MODE=none ZXJ_TARGET_POWER=1.0 python -m harness.loop --data public --ckpt checkpoint.npz
  # 只做通道内归一
  ZXJ_INPUT_MODE=abs8 ZXJ_TARGET_POWER=1.0 python -m harness.loop --data public --ckpt checkpoint.npz
  # 关掉集成（单轨迹）
  ENSEMBLE_BATCH= ZXJ_INPUT_MODE=abs8log ZXJ_TARGET_POWER=0.45 python -m harness.loop ...

决策闭环与门控规则见 harness/ 下各模块，规则阈值全部由环境变量给出，没有硬编码数字。

## 五、实测结果

  官方基线 RMSE              681.16
  常数（训练集中位数）        470.86
  P1（abs8 + p=0.5）          464.22（-31.9%）
  本方案 · 容器内             302.75（-55.6%）  MAE 230.41  末端 274.52
  本方案 · 开发机             313.79（-53.9%）  MAE 242.29  末端 276.66

两个环境的差异成因见 docs/P2_env_variance_check.md：数据、起点、环境变量逐位一致，
各自可复现，差异来自 BLAS 浮点归约顺序在 105 个 epoch 上的放大。提交口径取容器值。

数字出处：evidence/runs_image_v05/decisions.jsonl（0.5 版容器实跑）、
docs/P1_tail_fix_experiment.md、docs/P2_log_input_ensemble.md。

## 六、0.5 版修了什么（相对 0.4）

0.4 把数据目录写死在镜像里：Dockerfile 的 `ENV DATA_DIR=/app/public` 优先于 spec 里的
"/data 优先"探测，于是平台挂 /data 也会被忽略，日志里只看得到 `/app/public`。0.5 的改动：

1. 删掉两处 `ENV DATA_DIR`（Dockerfile / Dockerfile.slim），让代码里的探测生效；
2. spec 增加候选探测：/data、/data/public、/input、/input/public、/mnt/data、
   /workspace/data、/app/data，最后一个才是自带的 public/；
3. spec 事件新增 `data_source`（env / probe / bundled / fallback）与 `data_probes`
   （每个候选的 dir / train / validation 是否存在）。平台挂到非预期路径时，
   日志能直接列出"哪个目录存在但没有 train.jsonl"，据此设 DATA_DIR 重跑即可，不必重打镜像。

训练、门控、发布、数值一律未改：三轮仍是 302.75 / 305.99 / 342.01。

## 六之二、0.4 版修了什么（相对 0.3）

1. **检查点发布（P0）**：0.3 把最优权重写在 runs/round-01/checkpoint.npz，
   而 /app/checkpoint.npz 自始至终是官方起点。若平台按默认路径取模型，
   拿到的不是我们训出来的模型。0.4 在每次采纳后把权重原子写回
   /app/checkpoint.npz 与 /app/best_checkpoint.npz，summary.json 里写明
   submitted_checkpoint 与 submitted_checkpoint_sha256。
2. **起点备份（必须与第 1 条成对）**：启动时把原始 checkpoint.npz 备份成
   checkpoint.start.npz，后续轮次的重启目标指向备份。否则第 2 轮会把
   「权重平均点」当成续训起点，重新触发 P3 记录的崩坏（302.75 -> 352.7）。
3. **无验证集分支**：以前记 accept 却不推进 best_checkpoint；现在记
   advance_unverified，并真的推进，日志里不假装"有提升"。
4. **门控补两条**：单根轴承 RMSE 退化超过 25% 直接回滚（防止总体均值掩盖单根崩坏）；
   尾段样本少于 30 条时，末端权衡条款不适用。
5. **自检 8 项增到 10 项**：新增"发布与重启语义"与"决策日志容错"两条回归。

## 七、关于训练量的口径（避免误读）

训练入口的 epochs 是**每条轨迹**的轮数。ENSEMBLE_BATCH 配 6 条轨迹时，
提案里写 105 意味着实际训练 6 x 105 = 630 个 epoch。0.4 起：
  epochs_used        记提案轮数（255，可读性好）
  track_epochs_used  记真实工作量（1530 = 255 x 6）
环境变量 EPOCH_BUDGET=330 约束的是前者，即"每条轨迹的轮数预算"。
