"""The only door to an LLM: every model request goes to the Bifrost gateway.

What Bifrost does for this project:
  * one OpenAI-compatible endpoint for every provider (`provider/model` names)
  * a `fallbacks` array per request, so a provider outage does not stop a check
  * a virtual key (`x-bf-vk`) carrying a budget and rate limits
  * `x-bf-lh-*` headers stored as log metadata, which lets us rebuild cost per claim
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from .config import get_settings
from .ledger import Ledger

GOVERNANCE_HINTS = {
    401: "Bifrost rejected the virtual key (401). Check BIFROST_VIRTUAL_KEY.",
    402: "Bifrost budget exceeded (402). Raise the virtual key budget or wait for the reset.",
    403: "Bifrost blocked this model or provider for the virtual key (403).",
    429: "Bifrost rate limit hit (429). Slow down or raise the virtual key limits.",
}


class BifrostError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class LLMResult:
    content: str
    model_requested: str
    model_used: str
    provider: str
    is_fallback: bool
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    cache_hit: bool


def _headers(stage: str, run_id: str, claim_id: str, extra: dict | None = None) -> dict:
    s = get_settings()
    headers = {
        "Content-Type": "application/json",
        # Stored by Bifrost as log metadata (prefix stripped) -> per-claim ledger.
        "x-bf-lh-run-id": run_id,
        "x-bf-lh-stage": stage,
        "x-bf-lh-claim-id": f"{run_id}:{claim_id}",
    }
    if s.virtual_key:
        headers["x-bf-vk"] = s.virtual_key
    if extra:
        headers.update(extra)
    return headers


def _post(body: dict, headers: dict) -> dict:
    s = get_settings()
    try:
        resp = requests.post(f"{s.bifrost_url}/v1/chat/completions", json=body, headers=headers, timeout=s.timeout)
    except requests.RequestException as exc:
        raise BifrostError(0, f"Cannot reach Bifrost at {s.bifrost_url}. Is the gateway running? ({type(exc).__name__})") from exc
    if resp.status_code >= 400:
        hint = GOVERNANCE_HINTS.get(resp.status_code)
        detail = resp.text[:300]
        raise BifrostError(resp.status_code, f"{hint or 'Bifrost returned HTTP ' + str(resp.status_code)} {detail}".strip())
    try:
        return resp.json()
    except ValueError as exc:
        raise BifrostError(resp.status_code, "Bifrost returned a non-JSON response.") from exc


def chat(
    messages: list,
    *,
    model: str,
    stage: str,
    ledger: Ledger,
    claim_id: str = "",
    temperature: float = 0,
    cache_ttl: str | None = None,
) -> LLMResult:
    """One chat completion through Bifrost, recorded in the ledger."""
    s = get_settings()
    body = {"model": model, "messages": messages, "temperature": temperature}
    if s.fallbacks:
        body["fallbacks"] = list(s.fallbacks)  # Bifrost tries these when the primary fails

    extra = {}
    if cache_ttl is None:
        extra["x-bf-cache-no-store"] = "true"  # freshness checks must not be answered from cache
    else:  # only slow-moving claim types opt in, with a short TTL
        extra["x-bf-cache-key"] = f"bestbefore-{stage}"
        extra["x-bf-cache-ttl"] = cache_ttl

    claim_id = claim_id or stage  # stage-level calls (extract, rewrite) get their own id
    started = time.monotonic()
    data = _post(body, _headers(stage, ledger.run_id, claim_id, extra))
    latency_ms = int((time.monotonic() - started) * 1000)

    result = parse_completion(data, model, latency_ms)
    ledger.add(
        kind="llm",
        stage=stage,
        claim_id=claim_id,
        model_requested=model,
        model_used=result.model_used,
        provider=result.provider,
        is_fallback=result.is_fallback,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=latency_ms,
        cache_hit=result.cache_hit,
        cost_usd=None,  # filled from Bifrost logs by Ledger.enrich_from_bifrost
    )
    return result


def parse_completion(data: dict, model_requested: str, latency_ms: int = 0) -> LLMResult:
    try:
        content = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise BifrostError(200, "Bifrost response has no choices[0].message.content.") from exc
    if isinstance(content, list):  # content parts, e.g. [{"type": "text", "text": "..."}] from Gemini
        content = "".join(
            p.get("text") or "" if isinstance(p, dict) else str(p) for p in content
        )
    extra =data.get("extra_fields") or {}
    routing = extra.get("routing_info") or {}
    usage = data.get("usage") or {}
    cache = extra.get("cache_debug") or {}
    return LLMResult(
        content=content,
        model_requested=model_requested,
        model_used=str(routing.get("model") or data.get("model") or model_requested),
        provider=str(routing.get("provider") or extra.get("provider") or ""),
        is_fallback=bool(routing.get("is_fallback", False)),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        latency_ms=latency_ms,
        cache_hit=bool(cache.get("cache_hit", False)),
    )


def forward_chat(body: dict, ledger: Ledger) -> dict:
    """Proxy a client's chat request to Bifrost unchanged (non-streaming) and record it."""
    s = get_settings()
    body = dict(body)
    body["stream"] = False  # labels need the full answer
    if not body.get("model"):
        if not s.model_chat:
            raise BifrostError(400, "Request has no model and MODEL_CHAT is not set.")
        body["model"] = s.model_chat
    if s.fallbacks and "fallbacks" not in body:
        body["fallbacks"] = list(s.fallbacks)

    started = time.monotonic()
    data = _post(body, _headers("answer", ledger.run_id, "answer"))
    latency_ms = int((time.monotonic() - started) * 1000)
    result = parse_completion(data, body["model"], latency_ms)
    ledger.add(
        kind="llm", stage="answer", claim_id="answer", model_requested=body["model"], model_used=result.model_used,
        provider=result.provider, is_fallback=result.is_fallback, prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens, latency_ms=latency_ms, cache_hit=result.cache_hit, cost_usd=None,
    )
    return data
