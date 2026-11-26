# 智训引擎 · A/B/C 三题源代码包

赛题：A 气体六分类 / B 轴承 RUL（分钟）/ C C-MAPSS 涡扇 RUL（循环）
队伍：北京工商大学 · 智训引擎队
版本：R9 / 2026-11-26（镜像 zxj-harness:0.9，一个镜像三个提交位）

## 一、这个包里有什么

  baselines/        官方基线代码（model.py 按约定加入输入整形与目标变换）
  baselines_ckpt/   A / C 两题的官方起点；B 的起点在根目录 checkpoint.npz
  harness/          无人干预闭环：路由 -> 观察 -> 提案 -> 执行 -> 评估 -> 门控 -> 记录
  docs/             三题的模型配置、设计说明、历次实验记录
  evidence/         容器内实跑留档：A/B/C 各一份 decisions.jsonl 与 summary.json
  selfcheck.py      容器自检（12 项，含三题基线可加载与三题路由）
  pytrio_probe.py   大模型连通性探测（四步递进，纯标准库）

## 二、三题是怎么共用一个镜像的

判题只看数据的输入结构，不看环境变量、不看文件名：

  A  inputs.features 是长度 128 的数组            -> 六分类，Macro-F1
  B  inputs.features 是 {horizontal, vertical}    -> 回归（分钟），RMSE
  C  inputs.settings(3) + inputs.sensors(21)      -> 回归（循环），RMSE

路由发生在 harness/preflight.py，**必须在 import baselines 之前**完成，因为
baselines/model.py 在模块导入时就把 ZXJ_INPUT_MODE / ZXJ_TARGET_POWER 读成
模块级常量，之后再用 os.environ 改是无效的。这是本版最关键的一处结构约束，
也是为什么镜像入口是 preflight 而不是 loop。

## 三、复现

三题各自的单轮复现命令写在各题模型配置的第 [9] 节：

  docs/model_config_A.txt   A 题：12ep / lr0.005 / bs32    期望 macro_f1 = 0.7288
  docs/model_config_B.txt   B 题：105ep / lr0.03 / bs96×6  期望 rmse     = 302.75（容器内）
  docs/model_config_C.txt   C 题：30ep / lr0.003 / bs256   期望 rmse     = 51.2684

容器内跑完整闭环（无参数即用镜像内置默认值）：

  python -m harness.preflight

自检：

  python selfcheck.py

## 四、evidence 里是什么

  evidence/a/  6 轮：0.7288 采纳 -> 0.7267 / 0.7255 / 0.7252 / 0.7252 / 0.7252 回滚
  evidence/b/  2 轮：302.75 采纳 -> 305.99 回滚
  evidence/c/  4 轮：51.2684 采纳 -> 51.7159 / 51.1378 / 51.5772 回滚

每份都含 decisions.jsonl（逐轮观察/提案/结果/决策）与 summary.json（提交指纹、
轮次、耗时）。C 题第 3 轮的 51.1378 比提交值低 0.25% 但仍被回滚，原因是未达
0.5% 的改善门槛——这是规则的直接后果，已在模型配置第 [10] 节写明。

## 五、包里没有什么

不含隐藏测试数据、不含任何 API Key、不含平台凭证。B 题的 302.75 只在容器内成立
（Windows 本机同代码同数据为 315.92，差 4.3%，源于 OpenBLAS 归约顺序在 105 epoch
上的放大），因此复现请在 linux/amd64 + 2 核容器里做。
