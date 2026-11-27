# Harness 使用说明（B 题）

> **版本状态（2026-11-25）**：本次交付的镜像为 **`zxj-harness:0.8`**（tar 45,509,632 字节，
> sha256 `e816c4bd…`，自检 **10/10**，容器内 2 核非 root **两轮** 302.75 / 305.99，**314.6 秒**）。
> 0.8 把默认轮数从三轮改为两轮，并加了 `WALL_LIMIT_S=390` 的墙钟预算与自适应收尾——
> 平台窗口到点会强杀容器，自己收尾才能保证 summary 与权重先落盘；最终提交的一直是第 1 轮的
> 302.75，第 3 轮的 342.01 只是陪跑，删掉不影响成绩（见 `P7_wall_budget.md`）。
> 本文成文于 0.3–0.4 阶段，第四、十节里“0.4 / 8 项自检 / 89 秒”等表述是当时的历史记录，
> 数字与结论仍然成立，但**版本号与自检项数已过期**；0.5、0.6 的变更见
> `docs/P5_external_review_crosscheck.md`、`docs/P6_offline_llm_gate.md`。

面向企业赛：在平台的 B 题检查点上做**无人干预**的续训—评估—上线/回滚闭环，并留下可审计的决策链。

## 一、结构

```
pkg_b/
├─ baselines/            官方基线代码（model.py 已按约定改造输入整形）
├─ harness/
│  ├─ spec.py            路径/预算/门槛契约（全部来自环境变量）
│  ├─ observer.py        只看事实：数据规模、轴承分组、历史最好
│  ├─ planner.py         提方案：策略表优先，配了 PyTRiO 凭证才走大模型
│  ├─ policy.json        候选配置 + 本机实测先验
│  ├─ executor.py        调官方 train_run 跑一轮（拒绝从权重平均点续训）
│  ├─ evaluator.py       独立复评 + 末端 RMSE + 分轴承最差
│  ├─ gatekeeper.py      上线/回滚规则
│  ├─ recorder.py        decisions.jsonl（追加）+ state.json（fsync + 原子写）
│  ├─ publish.py         检查点发布：原子写回默认路径 + 起点备份
│  └─ loop.py            主循环
├─ selfcheck.py          容器自检（10 项，合成数据无依赖）
├─ pytrio_probe.py       PyTRiO 接入探测（纯标准库）
├─ mock_pytrio.py        本地模拟 PyTRiO，无凭证时验证链路
├─ Dockerfile            linux/amd64 镜像（Alpine 瘦身版，tar 40.5 MB）
├─ Dockerfile.slim       同内容 / Debian slim 基础镜像（tar 64.7 MB，仅备用）
├─ build_image.sh        构建 + 自检 + docker save + 经典格式导出
├─ to_classic_tar.py     OCI 布局转经典 Docker v2 布局
├─ checkpoint.npz        平台下发的起点权重
└─ public/               公开训练/验证集
```

## 二、本机已实测（可复跑）

| 项目 | 结果 |
|---|---|
| 官方单测 | 12 项中 10 项通过；2 项失败均为 `No module named 'scripts'`（发布包未含打包脚本，与改动无关） |
| `selfcheck.py`（合成 / 真实数据） | 10/10 通过 |
| 目标成绩（容器口径） | **RMSE 302.75 / MAE 230.41**；官方基线 681.16（**-55.6%**） |
| 输入整形 | abs8（通道内 8 个绝对量互为参照）+ 追加 14 列对数绝对量：34 维 → 48 维，参数 865 → 1201 |
| 目标变换 | y' = y^0.45；p ∈ [0.43, 0.47] 邻域内落在 303~391 |
| 多轨迹权重平均 | 6 条（96/112/128/144/192/256），单条 380.7~704.6，合并 **302.75** |
| 消融（开发机口径） | 34 维单轨迹 414.97 / 34 维 6 轨迹 515.74（更差）/ 48 维单轨迹 434.68 / 48 维 6 轨迹 313.79 |
| 容器内 3 轮闭环（0.4 版） | 轮 1 302.75 采纳；轮 2 305.99、轮 3 342.01 回滚；89 秒 |
| 镜像 | `zxj-harness:0.4`，linux/amd64，解压 189 MB，tar 40.5 MB，sha256 `e78dc4eb…` |
| 检查点发布（0.4 新增） | 采纳后 `/app/checkpoint.npz` 与 `best_checkpoint.npz` 同为 c8bd6608…；官方起点备份 c7d819fc… 保持不变 |
| rmi → load 还原 | 删除本地镜像后由 tar 重新导入并实跑通过 |
| 环境差异 | 同一配置：开发机 313.79（Py3.13/numpy2.5.4）、Py3.10 环境 307.74、容器 302.75；各环境内各自逐位复现 |
| **真实 PyTRiO（Qwen/Qwen3.8-27B）** | 探测 4/4 通过；多轮真跑 `source=llm`，269~322 prompt token、6~24 秒；模型给的理由是「先验 RMSE 最低」 |
| 重复提案防护 | 模型曾连提同一被回滚配置；加「已试过」记忆 + 代码层去重后每轮给出不同配置 |
| 降级分支 | JSON 模式被拒 → 去字段重试；`reasoning_effort` 被拒 → 去掉再试；回复非 JSON → 回退策略表且不中断 |

方法与对照见 `P2_集成与对数输入_实验记录.md`；闭环续训缺陷见 `P3_集成点不可续训_诊断.md`。

## 三、四条命令

```sh
# 0. 先确认大模型通不通（读 .env.pytrio，或显式 export）
python pytrio_probe.py

# 1. 本地只看不跑：观察 + 提案
python -m harness.loop --data public --ckpt checkpoint.npz --run-dir runs_dry --dry-run

# 2. 本地真跑（3 轮，330 epoch 预算）
python -m harness.loop --data public --ckpt checkpoint.npz --run-dir runs_demo --max-rounds 3 --epoch-budget 330

# 3. 容器自检 / 构建导出
python selfcheck.py
bash build_image.sh
```

## 四、环境变量契约

| 变量 | 默认 | 作用 |
|---|---|---|
| `DATA_DIR` | `/data`（无则 `/app/public`） | 含 `train.jsonl` / `validation.jsonl` |
| `CKPT` | `/app/checkpoint.npz` | 起点检查点 |
| `RUN_DIR` | `/app/runs` | 轮次输出、决策日志、状态 |
| `MAX_ROUNDS` | 2 | 最多几轮（0.8 起由 3 改 2，为的是压进平台窗口） |
| `EPOCH_BUDGET` | 330 | **每条轨迹**的轮数预算（105 × 3 轮量级）；6 轨迹时真实训练量 ×6，另记 `track_epochs_used` |
| `START_CKPT` | 与 `CKPT` 同目录的 `checkpoint.start.npz` | 官方起点备份；`publish` 覆盖 `CKPT` 后，续训重启指向它 |
| `MIN_TAIL_N` | 30 | 尾段样本少于此数时，末端权衡条款不适用 |
| `MAX_GROUP_DEGRADE` | 0.25 | 单根轴承 RMSE 退化上限，超过即回滚 |
| `MIN_GAIN` | 0.005 | 上线所需相对改善 |
| `TAIL_MINUTES` | 100 | 末端判定阈值（分钟） |
| `TAIL_TRADE_GAIN` | 0.20 | 末端权衡条款：末端至少改善多少才可换取总体退让 |
| `TAIL_TRADE_RMSE` | 0.06 | 末端权衡条款：总体 RMSE 允许退让的上限 |
| `WALL_LIMIT_S` | 390 | 墙钟预算（秒）。每轮开始前用「上一轮实测 × 1.2」估下一轮，装不进预算就收尾：先写 summary 再退 0，不等平台来杀 |
| `ZXJ_INPUT_MODE` | `abs8log` | `none` / `abs8` / `abs8log`；`none` 为复现官方行为 |
| `ZXJ_TARGET_POWER` | 0.45 | 目标变换 y'=y**p；`1.0` 为参考行为，`0.5` 回到第一版形态 |
| `ENSEMBLE_BATCH` | `96,112,128,144,192,256` | 集成轨迹的 batch_size 列表；留空退化为单轨迹训练 |
| `BASELINE_CPU_THREADS` | 2 | 训练线程数（平台限核时下调） |
| `PYTRiO_BASE_URL` | `https://pytrio.com/api/openai/v1` | OpenAI 兼容端点 |
| `PYTRiO_API_KEY` | 空 | 不配则 planner 100% 走策略表 |
| `PYTRiO_MODEL` | `Qwen/Qwen3.8-27B` | 基模名或自训权重路径 |
| `PYTRiO_MAX_TOKENS` | 1200 | 单次提案输出上限。27B 带思考链，给 200 会全耗在 thinking 上、content 返回空 |
| `PYTRiO_REASONING_EFFORT` | `low` | `low` / `medium`，不支持 `minimal`（422）。`low` 实测 token 减半且结论相同 |
| `ZXJ_USE_LLM` | `0` | 大模型提案总开关。**离线评测必须为 0**：凭证被平台注入也不调用，直接走策略表 |
| `PYTRiO_TIMEOUT` / `PYTRiO_RETRIES` | 8 / 1 | 超时秒数与重试次数（0.6 从 60/2 收紧：评测机不出网时，长超时会把运行窗口耗光） |

凭证可放在 `pkg_b/.env.pytrio`（已被 `.dockerignore` 排除），格式 `KEY=VALUE`，程序启动时自动载入；已存在的环境变量优先级更高，便于临时覆盖。**不要写进 `Dockerfile` 的 `ENV`**。

## 五、接入 PyTRiO（四步）

PyTRiO 是赛题指定的模型训推服务（情感机器），官方提供 **OpenAI 兼容接口**，因此不需要装 SDK、不需要 openai 库，用标准库 urllib 直连即可——这点对离线评测镜像很重要。

**第 1 步：拿 Key 和模型名**

| 要什么 | 从哪拿 |
|---|---|
| API Key | `pytrio.com` → 「总览」(dashboard) → 复制 API Key |
| base_url | 固定为 `https://pytrio.com/api/openai/v1` |
| 模型名 | 基模填 `Qwen/Qwen3.8-27B`；若用自己训的权重，填「权重」页里那条权重路径 |

**第 2 步：探测，先确认通再改代码**

```sh
export PYTRiO_BASE_URL=https://pytrio.com/api/openai/v1
export PYTRiO_API_KEY=你的Key
export PYTRiO_MODEL=Qwen/Qwen3.8-27B
python pytrio_probe.py
```

脚本依次验四件事，任一失败会直接告诉你卡在哪、返回码是什么：

| 检查 | 失败含义 | 退出码 |
|---|---|---|
| 配置齐全 | 缺 Key | 1 |
| `GET /models` 可访问 | Key 无效（401）或网络不通 | 2 |
| `chat/completions` 可用 | 协议不符，通常 base_url 少了 `/v1` | 3 |
| 输出是可解析 JSON | 模型不听话，需要换提示词或换模型 | 3 |

**第 3 步：跑一次带大模型的循环**

```sh
python -m harness.loop --data public --ckpt checkpoint.npz --run-dir runs_llm --max-rounds 2
```

日志里 `"source": "llm"` 就说明提案来自大模型；`llm.usage` 会记下这一轮消耗的 token，客观分统计的就是这个数。

**第 4 步：容器里注入，不要打进镜像**

```sh
docker run --rm \
  -e PYTRiO_BASE_URL=https://pytrio.com/api/openai/v1 \
  -e PYTRiO_API_KEY=你的Key \
  -e PYTRiO_MODEL=Qwen/Qwen3.8-27B \
  zxj-harness:0.3
```

Key 绝不能写进 `Dockerfile` 的 `ENV`——镜像会被提交和分发，写进去等于泄露。

**没有凭证时怎么验证这条链路？** 用本地 mock：

```sh
python mock_pytrio.py --port 8791          # 另开一个终端
PYTRiO_BASE_URL=http://127.0.0.1:8791/api/openai/v1 PYTRiO_API_KEY=k \
  python -m harness.loop --data public --ckpt checkpoint.npz --run-dir runs_llm --max-rounds 2
```

三种降级分支都已实测：服务端拒绝 `response_format` → 自动去掉该字段重试；模型返回非 JSON → 静默回退策略表；网络失败 → 重试后回退，循环不中断。

### 真机接入踩到的四个坑（都有实测依据）

| 现象 | 根因 | 处理 |
|---|---|---|
| HTTP 409 `insufficient_quota` | 账户余额为 0（Key 有效，`/models` 能通） | 充值，或换 `Qwen/Qwen3.5-4B`（单价约 1/5）；`pytrio_probe.py` 已单列退出码 4 |
| `content` 为 `null` | 27B 是推理型模型，`reasoning` 会吃掉 output 预算；实测一次 70–790 个 thinking token | 默认预算 1200；空内容时自动加倍重试（上限 2400）；探测脚本单独提示 |
| 同一配置反复提交 | 模型不知道哪些配置已被回滚；同配置从同一回滚点重训，结果逐位相同（实测连续 3 轮都是 486.01） | 提示词带「已试过 [lr,epochs,bs,RMSE,action]」；代码层再加一道去重，命中即转策略表推进下一个候选 |
| 单轮 token 忽高忽低（620 → 1298） | 提示词越长，thinking 越长 | 提示词改用数组省字段名（269 token）；`reasoning_effort=low` |

这几条不是理论推导，都是这次真机联调时先出问题、再改掉的。技术报告里可以直接写成「接入过程中发现并修复的工程问题」。

## 六、决策日志字段

`runs/decisions.jsonl` 每行一条：

```json
{"round":1,
 "ts":"...",
 "observation":{"best_rmse_before":null,"best_tail_before":null,"epochs_left":330},
 "proposal":{"lr":0.03,"epochs":105,"batch_size":96,"source":"policy","reason":"..."},
 "result":{"rmse":302.75,"mae":230.41,"tail_rmse":274.52,"tail_n":242,"worst_group":{...},"groups":{...}},
 "execution":{"checkpoint":"...","checkpoint_sha256":"...","parent_checkpoint_sha256":"...","requested_parent_sha256":"...","restarted_from_start":false,"initial_steps":825,"final_steps":6180},
 "decision":{"action":"accept","reason":"首个可评估轮次，直接作为当前最好（RMSE 302.75）"}}
```

`observation → proposal → result → decision` 四段齐全，即平台要求的「读历史—做实验—依据结果决策」完整记录。

## 七、上线/回滚规则（七条，可复述）

按顺序判定，命中即返回：

1. **无验证集** → 记 `advance_unverified`：推进 `best_checkpoint`，但 reason 里写明
   「本轮提升未经本地核实」。旧版记 `accept` 却不推进状态，日志与实际不一致。
2. **首个可评估轮** → 直接采纳，作为当前最好。
3. **单根轴承崩坏** → 任一轴承 RMSE 相对基准恶化超过 `MAX_GROUP_DEGRADE`（25%）即回滚，
   无论总体均值多好。依据：验证集 90% 样本来自同一根轴承，均值会掩盖单根退化。
4. **末端权衡条款** → 若总体 RMSE 退让 ≤6% 而末端 RMSE 改善 ≥20%，仍然采纳。依据是评分里「最差阶段与恢复速度」单独占分，不能只用总 RMSE 一条尺子。尾段样本少于 `MIN_TAIL_N`（30 条）时本条款不适用，避免一两条样本的偶然。
5. **未达门槛** → RMSE 改善不足 `MIN_GAIN`（0.5%）则回滚。
6. **涨分但末端崩** → 总体改善但末端恶化 >15%，同样回滚。
7. **其余** → 采纳。

**续训起点的额外约束**：若当前最优是 6 轨迹权重平均的产物（检查点带 `aggregate=weight_average`
标记），`executor` 不会从它继续训练，而是改从官方起点独立训练下一个候选，并在决策记录里写
`restarted_from_start: true` 与 `requested_parent_sha256`。原因：平均点的低误差来自 6 个函数
平均，不是权重落在好解上；从它续训，即使 lr 降到 1e-4、只跑 3 轮，也会退回 352.7（6 条轨迹
从 381~705 收敛到 352~354，彼此只差 3 分）。证据见 `P3_集成点不可续训_诊断.md`。

## 八、降级行为（重要）

平台若只挂载隐藏测试流、不提供带标签的验证集，`evaluator` 返回 `None`，`gatekeeper` 返回
**`advance_unverified`**：真的把 `best_checkpoint` 推进到本轮产物，同时在 reason 里写明
「本轮提升未经本地核实」。既不会伪造成绩，也不会因为缺标签而崩溃，更不会出现
「日志说接受了、状态却没动」的自相矛盾。

## 八之二、检查点发布（0.4 新增，防交付错模型）

平台没有公开"评测时从哪个路径取模型"的契约。0.4 起，每轮采纳后：

1. 把本轮最优权重**原子**写回 `CKPT`（默认 `/app/checkpoint.npz`）与同目录 `best_checkpoint.npz`，
   复制前后各算一次 sha256，不一致直接报错；
2. 启动时已把官方起点备份为 `checkpoint.start.npz`（已存在则不覆盖），
   后续轮次的重启目标指向备份——否则第 2 轮会把权重平均点当起点（见 P3）；
3. `summary.json` 写 `submitted_checkpoint` / `submitted_checkpoint_sha256` /
   `start_checkpoint_sha256`，无论平台读哪条路径，都能对上同一份权重。

## 九、与赛题规则的对应

| 规则要求 | 本实现 |
|---|---|
| 无人工干预完成全部阶段 | `loop.py` 一次性跑到底，异常也只记录并回滚，不中断 |
| ≥3 次完整决策记录 | `decisions.jsonl` 每轮一条，字段固定 |
| 材料齐全、代码可复现 | `selfcheck.py` 10 项、`docker save` 未压缩 tar |
| 不获取隐藏测试标签 | 只读 `train.jsonl` / `validation.jsonl`；隐藏集不进镜像 |
| 受控机制，防无限制迭代 | `MAX_ROUNDS` + `EPOCH_BUDGET` + `WALL_LIMIT_S` 三重上限 |
| 不改结构的硬约束 | 不动 `hidden` / `seed` / `paths` / `task`，`--resume` 校验全部通过 |

## 十、尚未验证的部分（不要当成已完成）

1. ~~镜像没有在本机构建过~~ **已解除（2026-11-20，11-23 重建）**：`bash build_image.sh` 构建成功，`linux/amd64`，容器内 `selfcheck` **8/8**。当前产物为 `zxj-harness-0.4.tar`（40.5 MB，OCI 布局，sha256 `e78dc4eb…`），已实测 `docker rmi` 后由 tar `load` 还原并运行，单轮 302.75。
2. ~~PyTRiO 真实凭证未接~~ **已解除（2026-11-21）**：真实 Key 接入，`pytrio_probe.py` 四步全通过；真实 5 轮运行全部 `source=llm`，单轮 567–646 token / 6.5s（当期为第一版方案，成绩 464.22）。
   仍然存在一个**规则层疑问**：正式评测声明是「离线的运行环境」，而调用大模型需要联网。**评测环境是否对 PyTRiO 端点放行白名单，必须向赛题组确认**；若不放行，镜像内只能走策略表，报告里要如实说明。
3. **平台数据挂载路径未知**：`DATA_DIR` 的默认值是按常见约定猜的（`/data` → `/app/public` 回退）。拿到平台说明后要改。
4. **「10 轮优化」的 epoch 口径未知**：默认 `EPOCH_BUDGET=330`（约 3 轮 × 105 epoch）。若平台给的是总 50 epoch，把 `EPOCH_BUDGET` 调到 45，并把 `policy.json` 首条候选改为 30 epoch、`ENSEMBLE_BATCH` 留空退化为单轨迹。
5. **平台对镜像标签/文件名的要求未知**：当前标签 `zxj-harness:0.4`，文件名 `zxj-harness-0.4.tar`。若平台要求特定命名，改 `build_image.sh` 的两个默认参数即可。
6. ~~镜像内尚未用真实凭证跑过~~ **已解除（2026-11-23）**：`docker run --rm --env-file .env.pytrio zxj-harness:0.3 -m harness.loop …` 在容器内真实调通 27B，2 轮 `source=llm`，token 685 / 1034，与宿主机逐位一致（同为一版方案的 464.22）。**注意用 `--env-file` 而非 `-e`**，避免 Key 出现在 shell 历史里。
7. **平台实际数据路径仍未验证**：容器内 `/app/public` 是公开集，平台可能挂到 `/data` 或其他位置。R1 试运行会暴露，届时改 `DATA_DIR` 或 Dockerfile 一行重建。

## 十一、镜像构建踩过的坑（供复现）

| 现象 | 原因 | 处理 |
|---|---|---|
| `pip` 报 `ReadTimeoutError` on `files.pythonhosted.org` | 境内直连 PyPI 超时 | Dockerfile 增加 `ARG PIP_INDEX`，默认清华源，并加 `--retries 8 --timeout 120` |
| `FromPlatformFlagConstDisallowed` 警告 | `FROM --platform=linux/amd64` 是常量 | 只是警告，产物仍是 linux/amd64。要去掉可改用 `--platform` 构参数 |
| `docker save` 产出 `blobs/` + `index.json` | Docker 29 默认 containerd 镜像存储 | 用 `to_classic_tar.py` 额外导出经典 v2 布局，两份都留着 |
| `-v $(pwd)/x:/app/x` 挂载无效，容器内报 `No such file` | **Git Bash（MSYS）把 `/c/Users/...` 转成 `C:\ on /\LenovoSoftstore\...`**，挂载源变成了一个不存在的路径，Docker 仍返回 0 | 加 `MSYS_NO_PATHCONV=1`，并直接写 Windows 风格路径：`MSYS_NO_PATHCONV=1 docker run -v "C:/.../runs_image:/app/runs_image" …`。同理 `--run-dir /tmp/x` 会被转成 `C:/Users/.../Temp/x`，容器内会新建出怪路径 |
