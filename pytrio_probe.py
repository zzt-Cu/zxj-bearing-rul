"""PyTRiO 接入探测：只用标准库，容器内可直接跑。

回答四个问题：
  1) 地址和 Key 有没有配、通不通；
  2) 这个 Key 能看见哪些模型，要用的 27B 在不在列表里；
  3) chat/completions 能否正常返回；
  4) 服务端是否接受 response_format=json_object（planner 依赖它拿严格 JSON）。

用法：
  export PYTRiO_BASE_URL=https://pytrio.com/api/openai/v1
  export PYTRiO_API_KEY=...
  export PYTRiO_MODEL=Qwen/Qwen3.8-27B
  python pytrio_probe.py

退出码：0 全部通过 / 1 缺配置 / 2 鉴权或网络失败 / 3 协议或格式不符 / 4 账户余额不足
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness.envfile import load as _load_env_file

_load_env_file()

DEFAULT_BASE = "https://pytrio.com/api/openai/v1"
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
PROBE_PROMPT = (
    '你在做轴承剩余寿命模型的续训调参。从候选里选一个并只输出 JSON：'
    '{"lr": 浮点, "epochs": 整数, "batch_size": 整数, "reason": "一句话"}\n'
    '候选：[{"lr":0.03,"epochs":45,"batch_size":96},{"lr":0.02,"epochs":30,"batch_size":128}]\n'
    '约束：lr 在 [0.0001,0.1]，epochs 不超过 150，batch_size 在 [16,512]。'
)


def _request(url, key, payload=None, timeout=45):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": "Bearer " + key}
    if data:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers,
                                     method="GET" if data is None else "POST")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
        return response.status, raw, time.monotonic() - started, None
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")[:500]
        return error.code, body, time.monotonic() - started, "HTTP %d: %s" % (error.code, body)
    except (urllib.error.URLError, OSError) as error:
        return None, "", time.monotonic() - started, "%s: %s" % (type(error).__name__, error)


def _ok(text):
    print("[OK] " + text)


def _fail(text):
    print("[X ] " + text)


def main(argv=None):
    parser = argparse.ArgumentParser(description="PyTRiO 接入探测")
    parser.add_argument("--base-url", default=os.environ.get("PYTRiO_BASE_URL", DEFAULT_BASE))
    parser.add_argument("--api-key", default=os.environ.get("PYTRiO_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("PYTRiO_MODEL", DEFAULT_MODEL))
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args(argv)

    base = args.base_url.rstrip("/")
    print("=" * 62)
    print("PyTRiO 接入探测")
    print("=" * 62)
    print("base_url : %s" % base)
    print("model    : %s" % args.model)
    print("api_key  : %s" % ("已配置（%d 字符）" % len(args.api_key) if args.api_key else "未配置"))
    print()

    if not args.api_key:
        _fail("缺少 API Key。在 pytrio.com 的「总览」页复制，再 export PYTRiO_API_KEY=...")
        return 1
    if not base.startswith("http"):
        _fail("base_url 必须以 http 开头，当前是 %r" % base)
        return 1

    # 1) 连通性与鉴权：GET /models
    status, raw, cost, error = _request(base + "/models", args.api_key, timeout=args.timeout)
    if error and status in (401, 403):
        _fail("鉴权失败（%s）。Key 无效或没有该服务的权限。" % status)
        return 2
    if error and status is None:
        _fail("网络不可达：%s" % error)
        return 2
    if status == 404:
        _fail("GET /models 返回 404。base_url 可能少了 /v1，当前是 %s" % base)
        return 3

    model_ids = []
    if status == 200:
        try:
            listing = json.loads(raw)
            model_ids = [item.get("id") for item in listing.get("data", []) if item.get("id")]
        except (ValueError, AttributeError):
            pass
        _ok("GET /models 可访问，返回 %d 个模型" % len(model_ids))
        if model_ids:
            hit = [m for m in model_ids if args.model.split("/")[-1].lower() in m.lower()]
            print("     列表含目标模型：%s" % (hit[:5] if hit else "未匹配到，仍按配置的 model 直接尝试"))
    else:
        _fail("GET /models 返回 %s：%s（不致命，继续测 chat）" % (status, error))

    # 2) 严格 JSON 模式
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": PROBE_PROMPT}],
        "temperature": 0,
        "max_tokens": 800,
        "response_format": {"type": "json_object"},
    }
    status, raw, cost, error = _request(base + "/chat/completions", args.api_key, payload, args.timeout)
    used_json_mode = True
    if error and status in (400, 422):
        print("     response_format 被拒绝（%s），改测普通模式" % status)
        used_json_mode = False
        payload.pop("response_format", None)
        status, raw, cost, error = _request(base + "/chat/completions", args.api_key, payload, args.timeout)

    if error and (status in (402, 409) or "insufficient" in str(error).lower()):
        _fail("账户余额不足（HTTP %s）。Key 本身有效，是 pytrio.com 账户没有可用额度。" % status)
        print("     处理：登录 pytrio.com 充值；或把 PYTRiO_MODEL 换成 Qwen/Qwen3.5-4B（单价约为 27B 的 1/5）。")
        return 4

    if error:
        _fail("chat/completions 失败：%s" % error)
        if status in (401, 403):
            return 2
        return 3

    try:
        body = json.loads(raw)
        choice = (body.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content")
        finish = choice.get("finish_reason")
    except (ValueError, KeyError, IndexError) as parse_error:
        _fail("响应不是预期的 chat 结构：%s；原文前 300 字：%s" % (parse_error, raw[:300]))
        return 3

    usage = body.get("usage") or {}
    _ok("chat/completions 可用（%.2fs，json_mode=%s）" % (cost, used_json_mode))
    print("     tokens：prompt %s / completion %s / total %s，finish_reason=%s"
          % (usage.get("prompt_tokens"), usage.get("completion_tokens"),
             usage.get("total_tokens"), finish))
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if reasoning:
        print("     其中思考链 %s token（Qwen3.8-27B 是推理型模型，预算给小会把正文挤空）" % reasoning)

    # 3) 输出能否解析成 planner 需要的 JSON
    if not content or not content.strip():
        _fail("content 为空（finish_reason=%s）。预算被思考链吃掉了，把 max_tokens 调大到 1200 再试。"
              % finish)
        return 3
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        _fail("回复里没有 JSON 对象，planner 会回退策略表。原文：%s" % content[:200])
        return 3
    try:
        proposal = json.loads(content[start:end + 1])
    except ValueError as value_error:
        _fail("JSON 解析失败（%s）。原文：%s" % (value_error, content[:200]))
        return 3
    missing = [k for k in ("lr", "epochs", "batch_size") if k not in proposal]
    if missing:
        _fail("JSON 缺字段 %s，planner 会判为无效并回退策略表。原文：%s" % (missing, content[:200]))
        return 3

    _ok("输出是可用 JSON：lr=%s epochs=%s batch_size=%s"
        % (proposal.get("lr"), proposal.get("epochs"), proposal.get("batch_size")))
    print("     理由：%s" % str(proposal.get("reason", ""))[:120])
    print()
    print("结论：可以接入。把下面三行写进容器环境变量即可（不要打进镜像）：")
    print("  PYTRiO_BASE_URL=%s" % base)
    print("  PYTRiO_API_KEY=<你的 Key>")
    print("  PYTRiO_MODEL=%s" % args.model)
    print("提示：客观分统计大模型 Token，planner 已设 max_tokens=800、temperature=0，实测单次约 380 token。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
