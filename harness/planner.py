"""Planner：提出本轮方案。默认策略表驱动；配了 PyTRiO 凭证才升级为模型提案。

设计取舍：大模型只负责「提参数 + 说理由」，不生成训练程序。原因不是懒，
是无人干预场景下 6 天内让模型从零写训练代码的失败率太高。这条要在技术报告里写明。

PyTRiO 接入：官方提供 OpenAI 兼容接口（base_url https://pytrio.com/api/openai/v1），
这里只用 urllib，不引入 openai 依赖，保证评测期离线镜像里也能跑。
"""
from pathlib import Path
import json
import os
import time
import urllib.error
import urllib.request

try:  # 本地凭证文件（镜像内不存在，是空操作）
    from .envfile import load as _load_env_file
except ImportError:  # 以脚本方式直接导入 harness/planner.py 时
    from envfile import load as _load_env_file

_load_env_file()

POLICY_PATH = Path(__file__).resolve().parent / "policy.json"

LR_RANGE = (1e-4, 1e-1)
BS_RANGE = (16, 512)

DEFAULT_BASE_URL = "https://pytrio.com/api/openai/v1"
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"

# 大模型调用闸门：默认关闭。凭证被平台注入不等于要调用——
# 0.6 之前的行为是「有 key 就调」，评测环境出不了网时每次决策要等满 60 s 超时，
# 三轮下来可以把平台 10 分钟的入口预检窗口吃光（2026-11-25 10:33 实测）。
# 需要演示大模型链路时显式设 ZXJ_USE_LLM=1；默认走策略表，与已备案的 302.75 一致。
_LLM_TRIPPED = False


def _load_policy(profile="B"):
    """读策略表。B 的候选在顶层（保持原样），A/C 在 profiles 段里覆盖。

    只有 candidates / reference / note 会被覆盖，grid 与 B 的实测记录始终保留，
    这样日志里既能看出这一轮用的是哪张表，也不会丢掉 B 的历史证据。
    """
    with POLICY_PATH.open(encoding="utf-8") as stream:
        data = json.load(stream)
    section = (data.get("profiles") or {}).get(profile)
    if not section:
        return data
    merged = dict(data)
    merged.update(section)
    return merged


def _clamp(value, low, high):
    return max(low, min(high, value))


def _valid(proposal, epochs_left):
    try:
        lr = float(proposal["lr"])
        epochs = int(proposal["epochs"])
        batch_size = int(proposal["batch_size"])
    except (KeyError, TypeError, ValueError):
        return None
    if not LR_RANGE[0] <= lr <= LR_RANGE[1]:
        return None
    if not BS_RANGE[0] <= batch_size <= BS_RANGE[1]:
        return None
    if not 1 <= epochs <= max(epochs_left, 1):
        return None
    return {"lr": lr, "epochs": epochs, "batch_size": batch_size}


def _describe(candidate):
    return "%dep / lr%g / bs%d" % (candidate["epochs"], candidate["lr"], candidate["batch_size"])


def _policy_proposal(facts, policy, epochs_left):
    candidates = policy["candidates"]
    rounds = int(facts.get("history_rounds", 0) or 0)
    # A/C 的历史分在 history_score 里，B 的 history_rmse 是同一个量的旧名字
    history = [value for value in (facts.get("history_score") or []) if value is not None]
    if not history:
        history = facts.get("history_rmse") or []
    is_b = facts.get("profile", "B") == "B"

    # 首轮：直接用本机实测最优的那一组，不做探索
    if rounds == 0 or not history:
        chosen = dict(candidates[0])
        prior = chosen.get("prior_rmse", chosen.get("prior"))
        return {"lr": chosen["lr"], "epochs": min(chosen["epochs"], epochs_left),
                "batch_size": chosen["batch_size"], "source": "policy",
                "reason": "首轮采用%s（%s，%s）"
                          % ("本机实测最优配置" if is_b else "该题实测最优配置",
                             _describe(chosen),
                             ("prior RMSE %s" % prior) if is_b else ("prior %s" % prior))}

    # 之后严格按候选表逐条推进。表内顺序已按实测量级排列，避免在轮次预算只有个位数时
    # 做大跨度跳跃；上轮被采纳时额外沿用上一轮的学习率并衰减，做一次局部细化。
    index = min(rounds, len(candidates) - 1)
    chosen = dict(candidates[index])
    best = facts.get("best_score") if not is_b else facts.get("best_rmse")
    if best is None:
        best = facts.get("best_rmse")
    last = history[-1]
    accepted = best is not None and last is not None and abs(float(last) - float(best)) < 1e-9
    if facts.get("best_is_aggregate"):
        # 当前最优是 6 条轨迹的权重平均点，从它续训必然劣化（302.75 起训 3 轮即 352+）。
        # 这一轮改为从官方起点独立训练一个候选，保持候选表里的学习率不变。
        reason = ("当前最优为集成平均点，不可续训；从官方起点独立训练候选 #%d（%s）"
                  % (index + 1, _describe(chosen)))
    elif accepted:
        chosen["lr"] = max(chosen["lr"] * 0.8, LR_RANGE[0])
        reason = "上轮被采纳，推进到候选 #%d（%s）并衰减学习率细化" % (index + 1, _describe(chosen))
    else:
        reason = "上轮未达门槛，推进到候选 #%d（%s）" % (index + 1, _describe(chosen))

    epochs = min(int(chosen["epochs"]), epochs_left)
    if epochs < 5:
        return None
    return {"lr": float(chosen["lr"]), "epochs": epochs,
            "batch_size": int(chosen["batch_size"]), "source": "policy", "reason": reason}


def _llm_settings():
    return {
        "base": os.environ.get("PYTRiO_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        "key": os.environ.get("PYTRiO_API_KEY", ""),
        "model": os.environ.get("PYTRiO_MODEL", DEFAULT_MODEL),
        "timeout": float(os.environ.get("PYTRiO_TIMEOUT", "8")),
        # Qwen3.8-27B 带思考链：实测一次决策 230–800 个 reasoning token（随提示变长而增长）。
        # 给 800 会偶发"content 为空→加倍重试"，那一轮实际花掉约 2000 token，比直接给 1200 更贵。
        "max_tokens": int(os.environ.get("PYTRiO_MAX_TOKENS", "1200")),
        # 思考链强度：minimal 服务端不支持（422），low 实测 230 token / 3s，
        # medium 400 token / 7.7s，且两者给出的配置相同。省 Token 就是省客观分。
        "effort": os.environ.get("PYTRiO_REASONING_EFFORT", "low").strip().lower(),
        "retries": max(1, int(os.environ.get("PYTRiO_RETRIES", "1"))),
        # 只有显式打开才调模型；默认关闭，等价于「无凭证」时的策略表路径
        "enabled": os.environ.get("ZXJ_USE_LLM", "0").strip().lower() in ("1", "true", "yes", "on"),
    }


def _llm_prompt(facts, policy, epochs_left):
    """压到最短：客观分按 Token 计费，提示里只放决策必需的数值。

    带上 prior_rmse：让模型基于我们本机实测过的结果选，而不是凭常识猜。
    不给它这些数，实测会给出 epochs=5 这种明显劣化的配置。
    """
    history = facts.get("history_rmse") or []
    reference = policy.get("reference") or {}
    # 用数组而不是字典：同样的信息，字段名全省掉，实测提示从 490 token 降到约 300。
    brief = {
        "基线": reference.get("official_baseline_rmse"),
        "上轮": None if not history else round(history[-1], 2),
        "最好": None if facts.get("best_rmse") is None else round(facts["best_rmse"], 2),
        "剩余epoch": epochs_left,
        "候选[lr,epochs,batch_size,先验RMSE]": [
            [c["lr"], c["epochs"], c["batch_size"], c.get("prior_rmse")]
            for c in policy["candidates"][:4]],
        "已试[lr,epochs,batch_size,RMSE,action]": [
            [t.get("lr"), t.get("epochs"), t.get("batch_size"), t.get("rmse"), t.get("action")]
            for t in (facts.get("history_trials") or [])[-2:]],
    }
    return ("轴承RUL续训，RMSE越低越好。选下一轮配置，只输出JSON："
            '{"lr":浮点,"epochs":整数,"batch_size":整数,"reason":"不超过20字"}。'
            "优先选先验RMSE低的；禁止重复已试里 action=reject 的配置；"
            "lr在[0.0001,0.1]，epochs不超过剩余epoch，batch_size在[16,512]。\n"
            + json.dumps(brief, ensure_ascii=False, separators=(",", ":")))


def _trial_key(item):
    if item.get("lr") is None or item.get("epochs") is None or item.get("batch_size") is None:
        return None
    return (round(float(item["lr"]), 8), int(item["epochs"]), int(item["batch_size"]))


def _repeats_rejected(proposal, trials):
    """提案是否与某个被回滚的配置完全相同。硬件代码层兜底，不只靠提示词。"""
    key = _trial_key(proposal)
    if key is None:
        return False
    return any(_trial_key(t) == key for t in trials if t.get("action") != "accept")


def _llm_post(settings, prompt, json_mode, use_effort=True):
    payload = {
        "model": settings["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": settings["max_tokens"],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if use_effort and settings.get("effort"):
        payload["reasoning_effort"] = settings["effort"]
    request = urllib.request.Request(
        settings["base"] + "/chat/completions", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + settings["key"]},
        method="POST")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=settings["timeout"]) as response:
            body = json.loads(response.read().decode("utf-8"))
        choice = (body.get("choices") or [{}])[0]
        return {"content": (choice.get("message") or {}).get("content"),
                "finish_reason": choice.get("finish_reason"),
                "usage": body.get("usage") or {},
                "latency_s": round(time.monotonic() - started, 3)}, None
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:300]
        return None, {"kind": "http", "status": error.code, "detail": detail}
    except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as error:
        return None, {"kind": "network", "detail": "%s: %s" % (type(error).__name__, error)}


def _llm_proposal(facts, policy, epochs_left):
    global _LLM_TRIPPED
    settings = _llm_settings()
    if not settings["key"]:
        return None
    if not settings["enabled"]:
        facts["_llm_error"] = {"kind": "disabled",
                               "detail": "未设 ZXJ_USE_LLM=1，本轮按策略表推进（离线评测默认行为）"}
        return None
    if _LLM_TRIPPED:
        facts["_llm_error"] = {"kind": "circuit_open",
                               "detail": "本次运行已出现过一次网络失败，后续轮次不再尝试大模型"}
        return None
    prompt = _llm_prompt(facts, policy, epochs_left)

    last_error = None
    budget = settings["max_tokens"]
    # 降级阶梯：严格 JSON+思考档 → 普通 JSON+思考档 → 普通 JSON 且不带思考参数
    # 平台若拒绝 response_format 或 reasoning_effort（400/422），逐级退，不整体放弃。
    variants = ((True, True), (False, True), (False, False))
    for attempt in range(settings["retries"]):
        for json_mode, use_effort in variants:
            result, error = _llm_post(dict(settings, max_tokens=budget), prompt, json_mode, use_effort)
            if error:
                last_error = error
                if error.get("kind") == "http" and error.get("status") in (400, 404, 422):
                    continue
                if error.get("kind") == "network":
                    # 网络不通换请求形状也没用，直接熔断，别让下一轮再等一次超时
                    _LLM_TRIPPED = True
                break
            text = result["content"] or ""
            if not text.strip():
                # 推理型模型把预算耗尽在 thinking 上时 content 为空；加预算再来，
                # 不能当成格式错误直接放弃，否则真机上会静默回退策略表。
                last_error = {"kind": "truncated",
                              "detail": "content 为空（finish_reason=%s，max_tokens=%d）"
                                        % (result.get("finish_reason"), budget)}
                budget = min(budget * 2, 2400)
                continue
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                last_error = {"kind": "format", "detail": "回复不含 JSON 对象"}
                continue
            try:
                proposal = json.loads(text[start:end + 1])
            except ValueError as value_error:
                last_error = {"kind": "format", "detail": "JSON 解析失败 %s" % value_error}
                continue
            checked = _valid(proposal, epochs_left)
            if not checked:
                last_error = {"kind": "range", "detail": "提案越界或字段缺失：%s" % json.dumps(proposal, ensure_ascii=False)[:200]}
                continue
            if _repeats_rejected(checked, facts.get("history_trials") or []):
                # 不重试，直接交给策略表推进到下一个未试候选：重试大概率还是同一个答案。
                facts["_llm_error"] = {"kind": "repeat",
                                       "detail": "提案与已回滚配置相同（%s），转由策略表推进候选"
                                                 % json.dumps(_trial_key(checked))}
                return None
            checked.update(source="llm",
                           reason=str(proposal.get("reason", ""))[:200] or "模型提案",
                           llm={"model": settings["model"], "json_mode": json_mode,
                                "reasoning_effort": settings["effort"] if use_effort else None,
                                "usage": result["usage"], "max_tokens": budget,
                                "reasoning_tokens": ((result["usage"].get("completion_tokens_details") or {})
                                                     .get("reasoning_tokens")),
                                "latency_s": result["latency_s"],
                                "attempt": attempt + 1})
            return checked
        time.sleep(1.5 * (attempt + 1))

    # 大模型不可用不阻塞流程：回退策略表，并把失败原因带进日志供审计
    facts["_llm_error"] = last_error
    return None


def propose(facts, policy=None):
    policy = dict(policy or _load_policy())
    epochs_left = int(facts.get("epochs_left", 0))
    if epochs_left < 5:
        return None
    for build in (_llm_proposal, _policy_proposal):
        proposal = build(facts, policy, epochs_left)
        if proposal:
            return proposal
    return None


def remember(proposal, profile="B"):
    """把本轮方案写回策略表内存，供下一轮自适应使用。

    必须带 profile：否则下一轮拿到的是 B 的候选表，A/C 会从第 2 轮起跳回
    「90ep / lr0.024 / bs96」这种为 B 设计、对 A/C 完全错误的配置。
    """
    policy = _load_policy(profile)
    policy["_last"] = {k: proposal[k] for k in ("lr", "epochs", "batch_size")}
    return policy
