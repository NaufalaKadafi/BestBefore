"""HTTP surface.

GET  /                      one-page UI
POST /api/check             {"answer": "..."} -> freshness report
POST /v1/chat/completions   drop-in proxy: forwards to Bifrost, then labels the answer
GET  /api/demo              sample answers
GET  /api/health            configuration and gateway status
"""

from __future__ import annotations

from pathlib import Path

import requests
from flask import Flask, jsonify, request, send_from_directory

from . import bifrost, pipeline
from .bifrost import BifrostError
from .config import ConfigError, get_settings, load_demo_answers
from .tavily import TavilyError

STATIC = Path(__file__).resolve().parent / "static"


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)

    @app.errorhandler(ConfigError)
    def _config(err):
        return jsonify(error={"type": "config", "message": str(err)}), 500

    @app.errorhandler(BifrostError)
    def _bifrost(err):
        status = err.status if 400 <= err.status < 600 else 502
        return jsonify(error={"type": "bifrost", "message": str(err), "status": err.status}), status

    @app.errorhandler(TavilyError)
    def _tavily(err):
        return jsonify(error={"type": "tavily", "message": str(err), "status": err.status}), 502

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/api/demo")
    def demo():
        return jsonify(load_demo_answers())

    @app.get("/api/health")
    def health():
        s = get_settings()
        gateway = "unreachable"
        try:
            requests.get(s.bifrost_url, timeout=3)
            gateway = "reachable"
        except requests.RequestException:
            pass
        return jsonify(
            bifrost_url=s.bifrost_url, bifrost=gateway, virtual_key=bool(s.virtual_key),
            model_extract=s.model_extract or None, model_judge=s.model_judge or None,
            fallbacks=list(s.fallbacks), tavily_key=bool(s.tavily_key),
        )

    @app.post("/api/check")
    def check():
        body = request.get_json(silent=True) or {}
        answer = str(body.get("answer") or "").strip()
        if not answer:
            return jsonify(error={"type": "input", "message": "Field 'answer' is required."}), 400
        ledger = pipeline.new_ledger()
        report = pipeline.check_answer(answer, ledger)
        if body.get("rewrite"):
            report["rewritten"] = pipeline.rewrite_answer(report, ledger)
            _refresh_ledger(report, ledger)
        return jsonify(report)

    @app.post("/v1/chat/completions")
    def proxy():
        """OpenAI-compatible. Header x-bb-mode: annotate (default) | rewrite | off."""
        body = request.get_json(silent=True) or {}
        if not body.get("messages"):
            return jsonify(error={"type": "input", "message": "Field 'messages' is required."}), 400
        mode = (request.headers.get("x-bb-mode") or "annotate").lower()
        ledger = pipeline.new_ledger()
        data = bifrost.forward_chat(body, ledger)
        if mode == "off":
            return jsonify(data)

        message = data["choices"][0]["message"]
        answer = message.get("content") or ""
        report = pipeline.check_answer(answer, ledger)
        if mode == "rewrite":
            rewritten = pipeline.rewrite_answer(report, ledger)
            if rewritten:
                report["rewritten"] = rewritten
                answer = rewritten
            _refresh_ledger(report, ledger)
        labels = pipeline.labels_text(report)
        message["content"] = answer + ("\n\n" + labels if labels else "")
        data["bestbefore"] = report  # structured labels for clients that want them
        return jsonify(data)

    return app


def _refresh_ledger(report: dict, ledger) -> None:
    s = get_settings()
    ledger.enrich_from_bifrost(s.bifrost_url, s.virtual_key)
    report["ledger"] = {"totals": ledger.totals(), "entries": ledger.entries, "notes": ledger.notes}
