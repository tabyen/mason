#!/usr/bin/env python3
"""Live test of the running gateway against your real AWS account. Costs a few cents.

    cd gateway && cp .env.example .env    # edit it
    docker compose up -d --build
    python3 tests/live_test.py            # reads .env; add --models gpt-5.6-luna,grok-4.7 to narrow

Each check sends a tiny request and prints PASS or FAIL with the gateway's error message.
Standard library only. Nothing here prints your keys.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

GATEWAY_DIR = Path(__file__).resolve().parent.parent
TOOL = {"type": "function", "function": {
    "name": "get_secret_word", "description": "Returns the secret word.",
    "parameters": {"type": "object", "properties": {}, "required": []}}}
results: list[tuple[str, str, str]] = []


def env_file() -> dict:
    out = {}
    path = GATEWAY_DIR / ".env"
    if path.is_file():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def call(url: str, body: dict | None = None, key: str = "", headers: dict | None = None, timeout: int = 180):
    """Returns (status, parsed JSON or text)."""
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {}), **(headers or {})}
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw, status = r.read().decode(), r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read().decode(errors="replace"), e.code
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def explain(status: int, data) -> str:
    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        data = err.get("message", err) if isinstance(err, dict) else err
    return f"HTTP {status}: {str(data)[:400]}"


def check(name: str, fn) -> None:
    start = time.time()
    try:
        ok, detail = fn()
    except Exception as e:  # a malformed reply is a failure worth reporting, not a crash
        ok, detail = False, f"{type(e).__name__}: {e}"
    results.append((name, "PASS" if ok else "FAIL", f"{detail}  ({time.time() - start:.1f}s)"))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma-separated model ids to test (default: everything in the catalog)")
    ap.add_argument("--effort", default="low", help="reasoning effort for the test requests (default: low)")
    ap.add_argument("--keep-going", action="store_true", help="run every check for a model even if its first one fails")
    a = ap.parse_args()
    env = {**env_file(), **os.environ}
    key = env.get("LITELLM_MASTER_KEY", "")
    gw = f"http://localhost:{env.get('GATEWAY_PORT', '4000')}"
    cat_url = f"http://localhost:{env.get('CATALOG_PORT', '8090')}"
    if not key:
        sys.exit("LITELLM_MASTER_KEY not found in gateway/.env or the environment.")

    print("Gateway and catalog")
    status, cat = call(f"{cat_url}/v1/catalog", timeout=15)
    catalog = cat.get("models", []) if isinstance(cat, dict) else []
    check("catalog reachable", lambda: (status == 200 and bool(catalog), f"{len(catalog)} models" if catalog else explain(status, cat)))
    status, listed = call(f"{gw}/v1/models", key=key, timeout=15)
    ids = [m["id"] for m in listed.get("data", [])] if isinstance(listed, dict) else []
    check("gateway reachable", lambda: (status == 200 and bool(ids), ", ".join(ids) if ids else explain(status, listed)))
    if not ids:
        print("\nThe gateway is not answering. Try: docker compose ps; docker compose logs litellm --tail 50")
        return 1
    missing = [m["id"] for m in catalog if m.get("available") is False]
    check("catalog and gateway agree", lambda: (not missing, "all catalog models are served" if not missing else f"not served: {missing}"))

    wanted = a.models.split(",") if a.models else ([m["id"] for m in catalog] or ids)
    by_id = {m["id"]: m for m in catalog}

    for model in wanted:
        entry = by_id.get(model, {})
        levels = (entry.get("effort") or {}).get("levels") or []
        effort = a.effort if a.effort in levels else None
        base = {"model": model, "max_tokens": 2000, **({"reasoning_effort": effort} if effort else {})}
        print(f"\n{model}  (upstream {(entry.get('upstream') or {}).get('model_id', '?')}, effort {effort or 'default'})")

        def chat():
            s, d = call(f"{gw}/v1/chat/completions", {**base, "messages": [
                {"role": "user", "content": "Reply with exactly the word: pong"}]}, key)
            if s != 200:
                return False, explain(s, d)
            text = (d["choices"][0]["message"].get("content") or "").strip()
            u = d.get("usage", {})
            return "pong" in text.lower(), f"{text[:40]!r}, {u.get('prompt_tokens')} in / {u.get('completion_tokens')} out"

        def tools():
            msgs = [{"role": "user", "content": "Call get_secret_word, then tell me the word it returned."}]
            s, d = call(f"{gw}/v1/chat/completions", {**base, "messages": msgs, "tools": [TOOL]}, key)
            if s != 200:
                return False, "turn 1: " + explain(s, d)
            m = d["choices"][0]["message"]
            if not m.get("tool_calls"):
                return False, f"turn 1: model did not call the tool: {(m.get('content') or '')[:100]!r}"
            msgs += [{"role": "assistant", "content": m.get("content"), "tool_calls": m["tool_calls"]},
                     {"role": "tool", "tool_call_id": m["tool_calls"][0]["id"], "content": "marzipan"}]
            s, d = call(f"{gw}/v1/chat/completions", {**base, "messages": msgs, "tools": [TOOL]}, key)
            if s != 200:
                return False, "turn 2 (tool result, reasoning carry-over): " + explain(s, d)
            text = d["choices"][0]["message"].get("content") or ""
            cached = (d.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens")
            return "marzipan" in text.lower(), f"{text[:50]!r}, cached tokens: {cached}"

        def stream():
            req = urllib.request.Request(f"{gw}/v1/chat/completions", data=json.dumps({
                **base, "stream": True, "messages": [{"role": "user", "content": "Count from 1 to 5."}]}).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            text, chunks = "", 0
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    for raw in r:
                        line = raw.decode().strip()
                        if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                            d = json.loads(line[5:])
                            if "error" in d:
                                return False, str(d["error"])[:300]
                            for c in d.get("choices", []):
                                text += c.get("delta", {}).get("content") or ""
                                chunks += 1
            except urllib.error.HTTPError as e:
                return False, explain(e.code, e.read().decode(errors="replace"))
            return "5" in text, f"{chunks} chunks, {text.strip()[:40]!r}"

        def responses():
            s, d = call(f"{gw}/v1/responses", {"model": model, "input": "Reply with exactly the word: pong",
                                               "max_output_tokens": 2000}, key)
            if s != 200:
                return False, explain(s, d)
            text = "".join(c.get("text", "") for o in d.get("output", []) if o.get("type") == "message"
                           for c in o.get("content", []))
            return "pong" in text.lower(), repr(text[:40])

        def messages():
            s, d = call(f"{gw}/v1/messages", {"model": model, "max_tokens": 2000, "messages": [
                {"role": "user", "content": "Reply with exactly the word: pong"}]}, key,
                headers={"anthropic-version": "2023-06-01"})
            if s != 200:
                return False, explain(s, d)
            text = "".join(c.get("text", "") for c in d.get("content", []) if c.get("type") == "text")
            return "pong" in text.lower(), repr(text[:40])

        check(f"{model} chat", chat)
        if results[-1][1] == "FAIL" and not a.keep_going:
            print("        (skipping the other checks for this model)")
            continue
        check(f"{model} tool call, two turns", tools)
        check(f"{model} streaming", stream)
        check(f"{model} responses format (Codex)", responses)
        check(f"{model} messages format (Claude Code)", messages)

    failed = [r for r in results if r[1] == "FAIL"]
    print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed")
    if failed:
        print("For details: docker compose logs mantle-translator --tail 40; docker compose logs litellm --tail 80")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
