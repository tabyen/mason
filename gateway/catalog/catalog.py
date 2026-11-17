#!/usr/bin/env python3
"""catalog: serves the gateway's model catalog (models, effort levels, pricing).

    GET /v1/catalog          the whole catalog as JSON
    GET /v1/catalog/{id}     one model
    GET /                    the same thing as an HTML table
    GET /health

Static facts come from models.toml (re-read when the file changes). `available` is live:
it is true when the gateway (LiteLLM) currently lists that model id, false when it does
not, and null when the gateway could not be asked. Standard library only.
"""
import html
import json
import os
import time
import tomllib
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DATA = os.environ.get("CATALOG_FILE", "/data/models.toml")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "").rstrip("/")     # e.g. http://litellm:4000
GATEWAY_KEY = os.environ.get("GATEWAY_KEY", "")
PORT = int(os.environ.get("PORT", "8090"))

_file = {"mtime": 0.0, "data": {}}
_live = {"at": 0.0, "ids": None}


def load() -> dict:
    mtime = os.stat(DATA).st_mtime
    if mtime != _file["mtime"]:
        with open(DATA, "rb") as f:
            _file["data"], _file["mtime"] = tomllib.load(f), mtime
    return _file["data"]


def live_ids() -> set | None:
    """Model ids the gateway serves right now, cached for 30 seconds."""
    if not GATEWAY_URL:
        return None
    if time.time() - _live["at"] > 30:
        try:
            req = urllib.request.Request(f"{GATEWAY_URL}/v1/models", headers={"Authorization": f"Bearer {GATEWAY_KEY}"})
            with urllib.request.urlopen(req, timeout=5) as r:
                _live["ids"] = {m["id"] for m in json.load(r).get("data", [])}
        except Exception:
            _live["ids"] = None
        _live["at"] = time.time()
    return _live["ids"]


def catalog() -> dict:
    data = load()
    ids = live_ids()
    models = []
    for m in data.get("models", []):
        pricing = {"currency": "USD", "unit": "1M tokens", "as_of": data.get("as_of"), **m.get("pricing", {})}
        models.append({**m, "pricing": pricing, "available": None if ids is None else m["id"] in ids})
    return {"schema": data.get("schema", "mason.catalog/v1"), "gateway": data.get("gateway", ""),
            "as_of": data.get("as_of"), "gateway_reachable": ids is not None,
            "unlisted": sorted(ids - {m["id"] for m in models}) if ids else [], "models": models}


def page(cat: dict) -> str:
    def money(v):
        return "" if v is None else f"${v:,.3f}".rstrip("0").rstrip(".")

    rows = []
    for m in cat["models"]:
        p, e = m["pricing"], m.get("effort", {})
        levels = ", ".join(f"<b>{l}</b>" if l == e.get("default") else l for l in e.get("levels", []))
        avail = {True: "yes", False: "NO", None: "?"}[m["available"]]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in [
            html.escape(m["id"]), html.escape(m.get("lab", "")), html.escape(m["upstream"]["model_id"]), avail,
            levels, f"{m.get('context_window', 0):,}" if m.get("context_window") else "",
            money(p.get("input")), money(p.get("output")), money(p.get("cache_read")),
            f'<a href="{html.escape(p.get("source", ""))}">source</a>']) + "</tr>")
    head = "".join(f"<th>{h}</th>" for h in ["model", "lab", "upstream id", "available", "effort (default bold)",
                                             "context", "input", "output", "cache read", ""])
    return (f"<!doctype html><meta charset=utf-8><title>Model catalog</title>"
            f"<style>body{{font:14px system-ui;margin:2rem}}table{{border-collapse:collapse}}"
            f"td,th{{border-bottom:1px solid #ccc;padding:.4rem .8rem;text-align:left}}</style>"
            f"<h1>Model catalog: {html.escape(cat['gateway'])}</h1>"
            f"<p>USD per 1M tokens, as of {html.escape(str(cat['as_of']))}. JSON at <a href=/v1/catalog>/v1/catalog</a>.</p>"
            f"<table><tr>{head}</tr>{''.join(rows)}</table>")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        try:
            if path == "/health":
                return self.send(200, b'{"status":"ok"}', "application/json")
            cat = catalog()
            if path == "":
                return self.send(200, page(cat).encode(), "text/html; charset=utf-8")
            if path == "/v1/catalog":
                return self.send(200, json.dumps(cat, indent=2).encode(), "application/json")
            if path.startswith("/v1/catalog/"):
                one = next((m for m in cat["models"] if m["id"] == path.rsplit("/", 1)[1]), None)
                if one:
                    return self.send(200, json.dumps(one, indent=2).encode(), "application/json")
            self.send(404, b'{"error":"not found"}', "application/json")
        except Exception as e:  # a bad models.toml should say so, not kill the server
            self.send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
