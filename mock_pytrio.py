"""本地 PyTRiO 模拟服务：只用于验证 planner 的模型提案路径是否真的能跑。

它按官方文档的 OpenAI 兼容形状响应：
  GET  {prefix}/models
  POST {prefix}/chat/completions

用途：没有真实凭证时，也要能证明「读事实 → 调大模型 → 解析 JSON → 落到决策日志」
这条链路是通的，而不是停留在代码没执行过。

用法：
  python mock_pytrio.py --port 8791 [--no-json-mode] [--bad-json]

  --no-json-mode  收到 response_format 就返回 400，用来测 planner 的降级分支
  --bad-json      返回一段非 JSON 文本，用来测解析失败后的回退
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"no_json_mode": False, "bad_json": False, "calls": []}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.path.endswith("/models"):
            return self._send(404, {"error": "not found"})
        if not self.headers.get("Authorization", "").startswith("Bearer "):
            return self._send(401, {"error": "missing key"})
        self._send(200, {"object": "list",
                         "data": [{"id": "Qwen/Qwen3.8-27B", "object": "model"}]})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        if not self.path.endswith("/chat/completions"):
            return self._send(404, {"error": "not found"})
        if not self.headers.get("Authorization", "").startswith("Bearer "):
            return self._send(401, {"error": "missing key"})
        try:
            request = json.loads(raw)
        except ValueError:
            return self._send(400, {"error": "bad json"})
        if STATE["no_json_mode"] and "response_format" in request:
            return self._send(400, {"error": "response_format not supported"})

        STATE["calls"].append({"model": request.get("model"),
                               "max_tokens": request.get("max_tokens"),
                               "messages": request.get("messages")})
        prompt = str(request.get("messages", [{}])[-1].get("content", ""))
        if STATE["bad_json"]:
            content = "我认为应该把学习率调小一点。"
        else:
            content = json.dumps({"lr": 0.024, "epochs": 22, "batch_size": 96,
                                  "reason": "上轮已达标，小幅衰减学习率再试一轮"},
                                 ensure_ascii=False)
        self._send(200, {
            "id": "mock-1", "object": "chat.completion",
            "model": request.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt) // 2,
                      "completion_tokens": len(content) // 2,
                      "total_tokens": (len(prompt) + len(content)) // 2},
        })


def main():
    parser = argparse.ArgumentParser(description="PyTRiO 本地模拟服务")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--bad-json", action="store_true")
    args = parser.parse_args()
    STATE["no_json_mode"] = args.no_json_mode
    STATE["bad_json"] = args.bad_json
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("mock PyTRiO 监听 127.0.0.1:%d  prefix=/api/openai/v1" % args.port, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
