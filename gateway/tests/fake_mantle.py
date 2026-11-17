#!/usr/bin/env python3
"""A scripted stand-in for Bedrock Mantle's Responses API (/openai/v1/responses).

Plays the same roles as ../../fake_llm.py, but speaks Responses, emits encrypted reasoning
items, and records what it was sent so tests can assert on it (GET /stats).
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
REVIEWED = '"""Calculator."""\n\n\ndef add(a, b):\n    """Return a + b."""\n    return a + b\n'
STATS = {"requests": 0, "reasoning_items_returned": 0, "bad_auth": 0, "efforts": [], "models": []}


def text_of(item) -> str:
    c = item.get("content")
    return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict))


def decide(body: dict):
    system = body.get("instructions", "")
    items = body.get("input") or []
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    users = [text_of(i) for i in items if i.get("role") == "user"] or [""]
    last = items[-1] if items else {}
    if "SCOUT" in system:
        return ("text", "calc.py does not exist yet; test_calc.py:1 imports calc.add.")
    if "PLANNER" in system:
        return ("text", "Plan: create calc.py with add(a, b).")
    if "CRITIC" in system:
        return ("text", "NO BLOCKING ISSUES")
    if "REVIEWER" in system:
        if '"""Return a + b."""' in users[0]:
            return ("text", '{"verdict": "approve", "issues": []}')
        return ("text", json.dumps({"verdict": "request_changes", "issues": [
            {"severity": "major", "file": "calc.py", "problem": "no docstring", "failing_case": "help(add)"}]}))
    if "DRIVER" in system:
        if last.get("type") == "function_call_output":
            return ("text", "Done: wrote calc.py.")
        if "independent reviewer" in users[-1]:
            return ("tool", "write_file", {"path": "calc.py", "content": REVIEWED})
        if "FAILED" in users[-1]:
            return ("tool", "write_file", {"path": "calc.py", "content": FIXED})
        return ("tool", "write_file", {"path": "calc.py", "content": BUGGY})
    return ("text", "ok")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._json(STATS)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not self.path.endswith("/openai/v1/responses"):
            return self._json({"error": {"message": f"unexpected path {self.path}"}}, 404)
        if not self.headers.get("Authorization", "").startswith("Bearer ") or \
                len(self.headers["Authorization"]) < 12:
            STATS["bad_auth"] += 1
            return self._json({"error": {"message": "missing bearer token"}}, 401)
        for key in ("messages", "max_tokens", "reasoning_effort", "output_config"):
            if key in body:
                return self._json({"error": {"message": f"unknown parameter: {key}"}}, 400)
        STATS["requests"] += 1
        STATS["models"].append(body.get("model"))
        STATS["efforts"].append((body.get("reasoning") or {}).get("effort"))
        items = body.get("input") if isinstance(body.get("input"), list) else []
        STATS["reasoning_items_returned"] += sum(1 for i in items if i.get("type") == "reasoning")
        # A reasoning item must be followed by the item it produced, as the real API requires.
        for a, b in zip(items, items[1:] + [{}]):
            if a.get("type") == "reasoning" and b.get("type") not in ("function_call", "message", "reasoning"):
                return self._json({"error": {"message": "reasoning item without its following item"}}, 400)

        d = decide(body)
        n = STATS["requests"]
        reasoning = {"type": "reasoning", "id": f"rs_{n}", "summary": [], "encrypted_content": f"enc-{n}"}
        if d[0] == "text":
            item = {"type": "message", "id": f"msg_{n}", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": d[1], "annotations": []}]}
        else:
            item = {"type": "function_call", "id": f"fc_{n}", "call_id": f"call_{n}", "name": d[1],
                    "arguments": json.dumps(d[2]), "status": "completed"}
        usage = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                 "input_tokens_details": {"cached_tokens": 40}, "output_tokens_details": {"reasoning_tokens": 8}}
        final = {"id": f"resp_{n}", "object": "response", "created_at": 1794889000, "status": "completed",
                 "model": body.get("model"), "output": [reasoning, item], "usage": usage, "error": None,
                 "incomplete_details": None, "instructions": body.get("instructions"), "metadata": {},
                 "parallel_tool_calls": True, "tool_choice": "auto", "tools": body.get("tools") or [],
                 "temperature": 1.0, "top_p": 1.0, "store": False}
        if not body.get("stream"):
            return self._json(final)

        events = [{"type": "response.created", "response": {**final, "status": "in_progress", "output": []}},
                  {"type": "response.output_item.added", "output_index": 0, "item": {**reasoning, "encrypted_content": None}},
                  {"type": "response.output_item.done", "output_index": 0, "item": reasoning}]
        if d[0] == "text":
            events += [{"type": "response.output_item.added", "output_index": 1, "item": {**item, "content": []}}]
            events += [{"type": "response.output_text.delta", "output_index": 1, "content_index": 0, "delta": piece}
                       for piece in (d[1][:len(d[1]) // 2], d[1][len(d[1]) // 2:])]
        else:
            args = item["arguments"]
            events += [{"type": "response.output_item.added", "output_index": 1, "item": {**item, "arguments": ""}}]
            events += [{"type": "response.function_call_arguments.delta", "output_index": 1, "delta": piece}
                       for piece in (args[:len(args) // 2], args[len(args) // 2:])]
        events += [{"type": "response.output_item.done", "output_index": 1, "item": item},
                   {"type": "response.completed", "response": final}]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for e in events:
            self.wfile.write(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode())


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4020
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
