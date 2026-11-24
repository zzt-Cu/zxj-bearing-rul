# P4 · 外部 AI 审阅核对（2026-11-24 10:00）

对象：外部模型对 `zxj-harness:0.3` 的静态审阅意见（6 条缺陷 + 1 段 P0 专项 + 3 处优先改）。
方法：逐条对照 `pkg_b/` 实际代码与容器内产物，不改代码、只核事实。

**结论：6 条里 5 条成立、1 条成立但定级过高；它给的 P0 修法会重新引入我们 P3 修掉的缺陷。**

---

## 一、逐条核对

| # | 它的定级 | 我的判定 | 修正后定级 | 依据 |
|---|---|---|---|---|
| 1 | P0 最优检查点未发布到 `/app/checkpoint.npz` | **成立**，与我 11-23 的审计一致 | **P0** | `loop.py:129-133` 只写 `state.json` 的路径；全仓只有 `executor.py:77` 生成 `runs/round-NN/checkpoint.npz`，`spec.ckpt` 自始至终只读不写 |
| 2 | P1 集成轨迹 6 倍 epoch 未计入预算 | **代码事实成立，定级过高** | **P2** | `train.py:105-142` 每条轨迹都跑 `epochs`；`loop.py:128` 只扣一次。但：① 算力无平台硬限制（`TASK.md` 只约束 3 轮）；② 实测单轮 31 s、全程 100 s，远低于 1500 s 墙钟；③ `training.json` 已用 `ensemble.batch_sizes` 披露 6 条轨迹 |
| 3 | P1 分轴承指标不进上线判定 | **成立** | **P2**（可留作改进，不影响本轮结果） | `evaluator.py:45-57` 算了 `groups`/`worst_group`，`gatekeeper.py` 全程只读 `rmse` 与 `tail_rmse`；`worst_group` 仅进日志 |
| 4 | P1 尾段 RMSE 缺最小样本量 | **成立** | **P3**（本数据 242 条，短期无风险） | `evaluator.py:40` 记了 `tail_n`，`gatekeeper.py:22-24` 未校验 |
| 5 | P1 无验证集时"记接受但不推进" | **成立，且比它说的更严重** | **P1（接近 P0）** | `gatekeeper.py:9-10` 直接 accept；`loop.py:129` 要求 `metrics is not None` 才更新，于是 `best_checkpoint` 永远停在起点。若平台不给 validation，`summary.best_checkpoint` 就是**起点本身**——与第 1 条叠加，等于交了官方起点 |
| 6 | P2 `state.json` 无 fsync、jsonl 可能残行 | **成立** | **P3** | `recorder.py:46-50` 无 flush+fsync、替换后不 fsync 目录；`decisions.jsonl` 单次 `write` 有 fsync，只有进程被杀在写一半时才会残行，且恢复只读 `state.json`，不影响运行 |

它引的行号与真实代码基本吻合（`executor.py:77-85`、`loop.py:129-133`、`evaluator.py:36-43`、`gatekeeper.py:9-15`、`recorder.py:24-31/43-50` 全部对得上），说明确实读了代码，不是编的。

---

## 二、它漏掉的两条（我们自己更该在意）

**L1｜`meta.epochs` 记的是 6 条轨迹的均值，不是工作量。**
`train.py:142` 把合并后检查点的 `epochs` 写成 `mean(各轨迹 epochs)`，`steps` 同理。所以
`round-01` 的 `final_epochs 120`、`final_steps 6180` 看上去等于"单轨迹跑了 105 epoch"，
实际是 630 epoch。它只说了"预算扣减"，没说"检查点自报的步数也是摊薄的"——
评审若拿 `final_steps - initial_steps` 核工作量，会得到我们少报了 6 倍。这条要写进文档，
不能靠 `ensemble.batch_sizes` 让读者自己推。

**L2｜墙钟只在轮次开头检查。**
`loop.py:79` 在每轮开始前判断，单轮内部无法中断。当前数据 31 s/轮，风险为零；
但如果隐藏集更大或平台限核更狠，1500 s 上限形同虚设。

---

## 三、它的 P0 修法有一个坑（必须改）

它建议在 accept 分支把最优检查点 `publish_checkpoint(state["best_checkpoint"], spec.ckpt)`。
**直接照抄会让我们重新踩 P3 的坑**：

```
executor.py:59-61
    if is_aggregate(parent_checkpoint):
        parent = str(spec.ckpt)     # ← 这里的 spec.ckpt 会被 publish 覆盖成"平均点"
        restarted = True
```

第 1 轮采纳的是**权重平均点**。一旦把它写回 `/app/checkpoint.npz`，第 2 轮
`is_aggregate(spec.ckpt)` 依然为真，于是 `parent = spec.ckpt` = 那个平均点本身——
P3 修掉的"从平均点续训即塌"（302.75 → 352.7）会原样回来，第 2 轮不再是 305.99。

**正确做法**（多一行、少一个坑）：

1. 启动时把原始起点备份一次到 `checkpoint.start.npz`（存在则跳过，不覆盖）；
2. `publish` 只写 `/app/checkpoint.npz` 与 `/app/best_checkpoint.npz`；
3. `executor` 的重启目标从 `spec.ckpt` 改成**备份那份起点**；
4. `summary` 里同时写 `submitted_checkpoint` + sha256 + `start_checkpoint_sha256`。

这样"平台读默认路径"和"第 2 轮从官方起点独立训练"两件事才不打架。

---

## 四、建议的 0.4 改动清单（按风险排序）

| 优先 | 改动 | 行为影响 | 风险 |
|---|---|---|---|
| 1 | `loop` 启动备份起点；accept 后原子发布到 `checkpoint.npz` + `best_checkpoint.npz`（带 sha 校验）；`executor` 重启改指向备份 | 主指标不变 | 低（必须与第 3 点一起改） |
| 2 | 无验证集分支：`action = advance_unverified`，推进 `best_checkpoint` 但 reason 写明"未经本地核实" | 有 validation 时零影响 | 低 |
| 3 | `summary` 补 `input_mode` / `ensemble` / `submitted_checkpoint` / `start_checkpoint_sha256` / `epochs_charged` | 无 | 低 |
| 4 | `gatekeeper` 加分轴承退化上限（如单根 RMSE 恶化 >25% 即拒）与 `tail_n ≥ 30` 才启用权衡条款 | 第 1 轮仍是首轮直接采纳；第 2/3 轮本就 reject，采纳集不变 | 低 |
| 5 | `recorder` 写 `state` 时 flush+fsync、替换后 fsync 目录；读 jsonl 容忍残行 | 无 | 低 |
| 6 | `selfcheck` 增两条回归：① 平均点必须重启自起点；② publish 前后 sha 一致 | 无 | 低 |
| 7 | 文档补 L1（epochs 是均值、6 条轨迹）与 L2（墙钟粒度） | 无 | 无 |

改完必须重建镜像 → 容器内复跑 3 轮，确认仍是 **302.75 / 305.99 / 342.01**、自检 10/10，
再连锁更新 tar sha256、运行说明、模型配置、提交清单。

---

## 五、一句话

它找出的问题是真的，但它给的 P0 补丁会踩到我们自己的 P3 结论——
**发布检查点这件事，必须和"起点备份"一起做**，否则第 2 轮的 305.99 会变成 352.7 那种结果。
