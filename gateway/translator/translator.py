#!/usr/bin/env python3
"""mantle-translator: an OpenAI-compatible front for OpenAI models on Amazon Bedrock Mantle.

What it does for the gateway in front of it (LiteLLM):

  * Auth. Turns AWS credentials (profile, env, role) into short-lived Bedrock bearer tokens,
    or uses a static Bedrock API key. Callers never hold AWS credentials.
  * Translation. Accepts Chat Completions and sends the Responses API upstream, converting
    messages, tools, tool calls, streaming events and usage both ways. On Mantle, prompt
    caching and reasoning summaries are Responses-only, so agent loops are cheaper this way.
  * Reasoning carry-over. Chat Completions has nowhere to put reasoning items, so the model
    would lose its reasoning between tool calls. The translator keeps the encrypted reasoning
    items in memory, keyed by tool-call id, and re-attaches them on the next request.
  * Pass-through. /v1/responses is forwarded as is (with auth and model mapping) for clients
    that speak Responses natively, such as Codex.

Configuration is by environment variable; see Settings below.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import OrderedDict

import aiohttp
from aiohttp import web

log = logging.getLogger("translator")


class Settings:
    def __init__(self) -> None:
        env = os.environ.get
        self.region = env("MANTLE_REGION") or env("AWS_REGION") or env("AWS_DEFAULT_REGION") or "us-east-1"
        # GPT-5.6 models live under /openai/v1 on Mantle (older gpt-oss models use /v1).
        self.base_url = (env("MANTLE_BASE_URL") or f"https://bedrock-mantle.{self.region}.api.aws/openai/v1").rstrip("/")
        self.static_key = env("MANTLE_API_KEY") or env("AWS_BEARER_TOKEN_BEDROCK") or ""
        self.inbound_key = env("TRANSLATOR_API_KEY", "")       # if set, callers must send it as a bearer token
        self.upstream_api = env("UPSTREAM_API", "responses")    # "responses" (translate) or "chat" (relay as is)
        self.carry_reasoning = env("CARRY_REASONING", "1") == "1"
        self.pass_sampling = env("PASS_SAMPLING", "0") == "1"   # forward temperature / top_p (reasoning models reject them)
        self.token_ttl = int(env("TOKEN_TTL_S", "900"))
        self.timeout = int(env("UPSTREAM_TIMEOUT_S", "900"))
        # alias -> upstream model id, e.g. {"gpt-5.6-terra": "openai.gpt-5.6-terra"}
        self.model_map: dict[str, str] = json.loads(env("MODEL_MAP", "{}"))
        self.port = int(env("PORT", "8080"))


S = Settings()

# ----------------------------------------------------------------------------- auth

_token = {"value": "", "at": 0.0}
_token_lock = asyncio.Lock()


async def upstream_token(force: bool = False) -> str:
    if S.static_key:
        return S.static_key
    async with _token_lock:
        if force or not _token["value"] or time.time() - _token["at"] > S.token_ttl:
            from aws_bedrock_token_generator import provide_token  # signs locally with the AWS credential chain
            _token["value"] = await asyncio.to_thread(provide_token, region=S.region)
            _token["at"] = time.time()
            log.info("minted a Bedrock bearer token for %s", S.region)
        return _token["value"]


# ----------------------------------------------------------------------------- reasoning carry-over

class ReasoningStore:
    """call_id -> reasoning items that preceded that tool call. Bounded, in memory."""

    def __init__(self, limit: int = 5000) -> None:
        self.items: OrderedDict[str, list] = OrderedDict()
        self.limit = limit

    def put(self, call_ids: list[str], reasoning: list) -> None:
        if not (S.carry_reasoning and call_ids and reasoning):
            return
        self.items[call_ids[0]] = reasoning
        while len(self.items) > self.limit:
            self.items.popitem(last=False)

    def get(self, call_id: str) -> list:
        return self.items.get(call_id, []) if S.carry_reasoning else []


STORE = ReasoningStore()

# ----------------------------------------------------------------------------- chat -> responses

def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if isinstance(p, dict))


def _user_parts(content) -> list:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    parts = []
    for p in content or []:
        if p.get("type") == "text":
            parts.append({"type": "input_text", "text": p.get("text", "")})
        elif p.get("type") == "image_url":
            img = p.get("image_url") or {}
            part = {"type": "input_image", "image_url": img.get("url") if isinstance(img, dict) else img}
            if isinstance(img, dict) and img.get("detail"):
                part["detail"] = img["detail"]
            parts.append(part)
        elif p.get("type") == "file":
            parts.append({"type": "input_file", **(p.get("file") or {})})
    return parts


def chat_to_responses(body: dict) -> dict:
    messages = body.get("messages") or []
    instructions, items, leading = [], [], True
    for m in messages:
        role = m.get("role")
        if role in ("system", "developer"):
            if leading:
                instructions.append(_text(m.get("content")))
            else:
                items.append({"role": "developer", "content": [{"type": "input_text", "text": _text(m.get("content"))}]})
            continue
        leading = False
        if role == "user":
            items.append({"role": "user", "content": _user_parts(m.get("content"))})
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            if calls:
                items.extend(STORE.get(calls[0].get("id", "")))
            text = _text(m.get("content"))
            if text:
                items.append({"type": "message", "role": "assistant",
                              "content": [{"type": "output_text", "text": text}]})
            for c in calls:
                fn = c.get("function") or {}
                items.append({"type": "function_call", "call_id": c.get("id"), "name": fn.get("name"),
                              "arguments": fn.get("arguments") or "{}"})
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": m.get("tool_call_id"),
                          "output": _text(m.get("content"))})

    out: dict = {"model": S.model_map.get(body.get("model", ""), body.get("model")), "input": items,
                 "store": False, "stream": bool(body.get("stream"))}
    if instructions:
        out["instructions"] = "\n\n".join(instructions)
    if S.carry_reasoning:
        out["include"] = ["reasoning.encrypted_content"]

    tools = []
    for t in body.get("tools") or []:
        if t.get("type") == "function":
            fn = t.get("function") or {}
            tool = {"type": "function", "name": fn.get("name"), "description": fn.get("description", ""),
                    "parameters": fn.get("parameters") or {"type": "object", "properties": {}}}
            if "strict" in fn:
                tool["strict"] = fn["strict"]
            tools.append(tool)
    if tools:
        out["tools"] = tools
        choice = body.get("tool_choice")
        if isinstance(choice, dict) and choice.get("type") == "function":
            out["tool_choice"] = {"type": "function", "name": (choice.get("function") or {}).get("name")}
        elif isinstance(choice, str):
            out["tool_choice"] = choice
        if "parallel_tool_calls" in body:
            out["parallel_tool_calls"] = body["parallel_tool_calls"]

    effort = body.get("reasoning_effort") or (body.get("reasoning") or {}).get("effort")
    if effort:
        out["reasoning"] = {"effort": effort}
    limit = body.get("max_completion_tokens") or body.get("max_tokens")
    if limit:
        out["max_output_tokens"] = limit
    fmt = body.get("response_format") or {}
    if fmt.get("type") == "json_schema":
        js = fmt.get("json_schema") or {}
        out["text"] = {"format": {"type": "json_schema", "name": js.get("name", "response"),
                                  "schema": js.get("schema", {}), "strict": js.get("strict", False)}}
    elif fmt.get("type") == "json_object":
        out["text"] = {"format": {"type": "json_object"}}
    if S.pass_sampling:
        for k in ("temperature", "top_p"):
            if body.get(k) is not None:
                out[k] = body[k]
    for k in ("prompt_cache_key", "service_tier"):
        if body.get(k) is not None:
            out[k] = body[k]
    return out


# ----------------------------------------------------------------------------- responses -> chat

def usage_to_chat(u: dict | None) -> dict:
    u = u or {}
    return {
        "prompt_tokens": u.get("input_tokens", 0), "completion_tokens": u.get("output_tokens", 0),
        "total_tokens": u.get("total_tokens", u.get("input_tokens", 0) + u.get("output_tokens", 0)),
        "prompt_tokens_details": {"cached_tokens": (u.get("input_tokens_details") or {}).get("cached_tokens", 0)},
        "completion_tokens_details": {
            "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)},
    }


def finish_reason(resp: dict, has_calls: bool) -> str:
    if has_calls:
        return "tool_calls"
    if (resp.get("incomplete_details") or {}).get("reason") == "max_output_tokens":
        return "length"
    if (resp.get("incomplete_details") or {}).get("reason") == "content_filter":
        return "content_filter"
    return "stop"


def responses_to_chat(resp: dict, model: str) -> dict:
    text, calls, reasoning = [], [], []
    for item in resp.get("output") or []:
        kind = item.get("type")
        if kind == "message":
            for part in item.get("content") or []:
                text.append(part.get("text") or part.get("refusal") or "")
        elif kind == "function_call":
            calls.append({"id": item.get("call_id"), "type": "function",
                          "function": {"name": item.get("name"), "arguments": item.get("arguments") or "{}"}})
        elif kind == "reasoning":
            reasoning.append(item)
    STORE.put([c["id"] for c in calls], reasoning)
    message: dict = {"role": "assistant", "content": "".join(text) or None}
    if calls:
        message["tool_calls"] = calls
    return {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
            "model": model, "choices": [{"index": 0, "message": message,
                                         "finish_reason": finish_reason(resp, bool(calls))}],
            "usage": usage_to_chat(resp.get("usage"))}


async def sse_events(stream: aiohttp.StreamReader):
    """Yield the JSON payload of each `data:` line of a server-sent event stream."""
    async for raw in stream:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    log.warning("unparseable event from upstream: %.200s", data)


async def stream_responses_as_chat(upstream: aiohttp.ClientResponse, request: web.Request, model: str,
                                   want_usage: bool) -> web.StreamResponse:
    out = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
    await out.prepare(request)
    cid, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())

    async def send(delta: dict | None = None, finish: str | None = None, usage: dict | None = None,
                   error: dict | None = None) -> None:
        if error is not None:
            payload: dict = {"error": error}
        else:
            payload = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                       "choices": [] if usage is not None else
                       [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
            if usage is not None:
                payload["usage"] = usage
        await out.write(f"data: {json.dumps(payload)}\n\n".encode())

    await send({"role": "assistant", "content": ""})
    tool_index: dict[int, int] = {}   # upstream output_index -> chat tool_calls index
    call_ids, reasoning, done = [], [], False
    async for ev in sse_events(upstream.content):
        kind = ev.get("type", "")
        if kind == "response.output_text.delta" or kind == "response.refusal.delta":
            await send({"content": ev.get("delta", "")})
        elif kind == "response.output_item.added" and (ev.get("item") or {}).get("type") == "function_call":
            item = ev["item"]
            n = tool_index[ev.get("output_index", len(tool_index))] = len(tool_index)
            call_ids.append(item.get("call_id"))
            await send({"tool_calls": [{"index": n, "id": item.get("call_id"), "type": "function",
                                        "function": {"name": item.get("name"), "arguments": ""}}]})
        elif kind == "response.function_call_arguments.delta":
            n = tool_index.get(ev.get("output_index", -1))
            if n is not None:
                await send({"tool_calls": [{"index": n, "function": {"arguments": ev.get("delta", "")}}]})
        elif kind == "response.output_item.done" and (ev.get("item") or {}).get("type") == "reasoning":
            reasoning.append(ev["item"])
        elif kind in ("response.completed", "response.incomplete"):
            resp = ev.get("response") or {}
            STORE.put(call_ids, reasoning)
            await send({}, finish=finish_reason(resp, bool(call_ids)))
            if want_usage:
                await send(usage=usage_to_chat(resp.get("usage")))
            done = True
        elif kind in ("response.failed", "error"):
            err = (ev.get("response") or {}).get("error") or ev.get("error") or ev
            await send(error={"message": str(err.get("message", err)), "type": "upstream_error",
                              "code": err.get("code")})
            done = True
    if not done:
        await send(error={"message": "upstream stream ended before completion", "type": "upstream_error"})
    await out.write(b"data: [DONE]\n\n")
    await out.write_eof()
    return out


# ----------------------------------------------------------------------------- http handlers

def openai_error(status: int, message: str, kind: str = "invalid_request_error") -> web.Response:
    return web.json_response({"error": {"message": message, "type": kind, "code": status}}, status=status)


def authorized(request: web.Request) -> bool:
    if not S.inbound_key:
        return True
    sent = request.headers.get("Authorization", "").removeprefix("Bearer ").strip() or request.headers.get("x-api-key", "")
    return sent == S.inbound_key


async def call_upstream(request: web.Request, path: str, payload: dict) -> aiohttp.ClientResponse:
    """POST to Mantle; on 401/403 mint a fresh token and try once more."""
    session: aiohttp.ClientSession = request.app["session"]
    for attempt in (1, 2):
        headers = {"Authorization": f"Bearer {await upstream_token(force=attempt == 2)}",
                   "Content-Type": "application/json"}
        resp = await session.post(f"{S.base_url}{path}", json=payload, headers=headers)
        if resp.status in (401, 403) and attempt == 1 and not S.static_key:
            resp.release()
            continue
        return resp
    raise AssertionError("unreachable")


async def relay_error(resp: aiohttp.ClientResponse) -> web.Response:
    text = await resp.text()
    log.warning("upstream %s: %.500s", resp.status, text)
    try:
        return web.json_response(json.loads(text), status=resp.status)
    except json.JSONDecodeError:
        return openai_error(resp.status, text[:2000] or "upstream error", "upstream_error")


async def relay(request: web.Request, path: str, body: dict) -> web.StreamResponse:
    """Forward a request unchanged except for auth and the model id."""
    body = {**body, "model": S.model_map.get(body.get("model", ""), body.get("model"))}
    resp = await call_upstream(request, path, body)
    async with resp:
        if resp.status >= 400:
            return await relay_error(resp)
        if not body.get("stream"):
            return web.json_response(await resp.json(content_type=None), status=resp.status)
        out = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await out.prepare(request)
        async for chunk in resp.content.iter_any():
            await out.write(chunk)
        await out.write_eof()
        return out


async def chat_completions(request: web.Request) -> web.StreamResponse:
    if not authorized(request):
        return openai_error(401, "invalid translator key", "authentication_error")
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return openai_error(400, "request body is not JSON")
    try:
        if S.upstream_api == "chat":
            return await relay(request, "/chat/completions", body)
        model = body.get("model", "")
        resp = await call_upstream(request, "/responses", chat_to_responses(body))
        async with resp:
            if resp.status >= 400:
                return await relay_error(resp)
            if body.get("stream"):
                want_usage = bool((body.get("stream_options") or {}).get("include_usage"))
                return await stream_responses_as_chat(resp, request, model, want_usage)
            return web.json_response(responses_to_chat(await resp.json(content_type=None), model))
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return openai_error(502, f"could not reach Bedrock Mantle: {e}", "upstream_error")
    except (RuntimeError, ValueError) as e:  # no AWS credentials, bad region
        return openai_error(500, f"translator auth/config error: {e}", "server_error")


async def responses(request: web.Request) -> web.StreamResponse:
    if not authorized(request):
        return openai_error(401, "invalid translator key", "authentication_error")
    try:
        return await relay(request, "/responses", await request.json())
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return openai_error(502, f"could not reach Bedrock Mantle: {e}", "upstream_error")
    except (RuntimeError, ValueError, json.JSONDecodeError) as e:
        return openai_error(500, f"translator error: {e}", "server_error")


async def models(request: web.Request) -> web.Response:
    ids = list(S.model_map) or []
    return web.json_response({"object": "list", "data": [{"id": i, "object": "model", "owned_by": "bedrock-mantle"}
                                                         for i in ids]})


async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "upstream": S.base_url, "upstream_api": S.upstream_api,
                              "auth": "static key" if S.static_key else "aws credential chain",
                              "carry_reasoning": S.carry_reasoning, "reasoning_entries": len(STORE.items)})


async def make_app() -> web.Application:
    app = web.Application(client_max_size=64 * 1024 ** 2)

    async def session_ctx(app: web.Application):
        app["session"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=S.timeout, sock_read=S.timeout))
        yield
        await app["session"].close()

    app.cleanup_ctx.append(session_ctx)
    for prefix in ("/v1", ""):
        app.router.add_post(f"{prefix}/chat/completions", chat_completions)
        app.router.add_post(f"{prefix}/responses", responses)
        app.router.add_get(f"{prefix}/models", models)
    app.router.add_get("/health", health)
    return app


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(name)s %(message)s")
    log.info("upstream=%s api=%s auth=%s", S.base_url, S.upstream_api,
             "static key" if S.static_key else "aws credential chain")
    web.run_app(make_app(), port=S.port, print=None)
