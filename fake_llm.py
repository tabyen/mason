#!/usr/bin/env python3
"""A scripted OpenAI-compatible server for smoke-testing crosslab without spending tokens.

It plays every role just well enough to exercise the whole pipeline: the driver first writes
a buggy calc.py (checks fail), fixes it when shown the failure, and addresses one review
finding; the reviewer objects once and then approves.

    python3 fake_llm.py 4010 &
    LITELLM_API_KEY=x python3 crosslab.py "add calc.add" --repo /tmp/scratch --config smoke.toml
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
REVIEWED = '"""Calculator."""\n\n\ndef add(a, b):\n    """Return a + b."""\n    return a + b\n'


def text_of(msg) -> str:
    c = msg.get("content") or ""
    return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c if isinstance(p, dict))


def decide(body: dict):
    """Return ("text", str) or ("tool", name, args)."""
    msgs = body["messages"]
    system = next((text_of(m) for m in msgs if m["role"] == "system"), "")
    last = msgs[-1]
    users = [text_of(m) for m in msgs if m["role"] == "user"]
    if "SCOUT" in system:
        return ("text", "calc.py does not exist yet; test_calc.py:1 imports calc.add. Check: python3 test_calc.py")
    if "PLANNER" in system:
        return ("text", "Plan: create calc.py with add(a, b). Test with test_calc.py.")
    if "CRITIC" in system:
        return ("text", "1. BLOCKING: the plan does not say what add returns for floats. State it.")
    if "REVIEWER" in system:
        if '"""Return a + b."""' in users[0]:
            return ("text", 'Looks right.\n```json\n{"verdict": "approve", "issues": []}\n```')
        return ("text", json.dumps({"verdict": "request_changes", "issues": [
            {"severity": "major", "file": "calc.py", "problem": "add has no docstring",
             "failing_case": "help(calc.add) shows nothing"}]}))
    if "DRIVER" in system:
        if last["role"] == "tool":
            return ("text", "Done: wrote calc.py.")
        if os.environ.get("FAKE_BAD_MODEL", "<unset>") in body["model"]:  # a driver that never gets it right
            return ("tool", "write_file", {"path": "calc.py", "content": BUGGY})
        if "independent reviewer" in users[-1]:
            return ("tool", "write_file", {"path": "calc.py", "content": REVIEWED})
        if "FAILED" in users[-1]:
            return ("tool", "write_file", {"path": "calc.py", "content": FIXED})
        content = BUGGY if "terra" in body["model"] else FIXED
        return ("tool", "write_file", {"path": "calc.py", "content": content})
    return ("text", "ok")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        d = decide(body)
        if d[0] == "text":
            deltas, finish = [{"role": "assistant", "content": d[1]}], "stop"
        else:
            deltas, finish = [{"role": "assistant", "tool_calls": [{
                "index": 0, "id": f"call_{len(body['messages'])}", "type": "function",
                "function": {"name": d[1], "arguments": json.dumps(d[2])}}]}], "tool_calls"
        base = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 0, "model": body["model"]}
        chunks = [{**base, "choices": [{"index": 0, "delta": x, "finish_reason": None}]} for x in deltas]
        chunks.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        chunks.append({**base, "choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for c in chunks:
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            msg = {"role": "assistant", "content": deltas[0].get("content"), "tool_calls": deltas[0].get("tool_calls")}
            out = {**base, "object": "chat.completion", "usage": chunks[-1]["usage"],
                   "choices": [{"index": 0, "message": msg, "finish_reason": finish}]}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4010
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
