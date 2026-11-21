"""Live web evidence from Tavily. This is the only source of evidence in the project.

Freshness-specific choices (all from the Tavily Search/Extract references):
  * max_age_hours=0 forces a live fetch for continuously updated pages (deprecation
    tables, pricing). Tavily's docs warn that time_range drops such pages.
  * include_domains + include_domains_mode="prefer" boosts official sources without
    hiding the wider web.
  * exact_match pins identifiers such as model IDs or names (quoted in the query).
  * include_published_date returns Tavily's date estimate, which can be null. Extract
    returns no date at all, so the date is carried forward from Search.

Free-tier behaviour handled here:
  * HTTP 434 (flex capacity) when max_age_hours is set -> retry without it and mark
    the evidence as possibly cached.
  * HTTP 400/422 (a parameter this account or API version rejects) -> retry once with
    core parameters only and record which strategy was lost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests

from .config import get_settings
from .ledger import Ledger

CORE_PARAMS = ("query", "max_results", "search_depth", "topic", "time_range", "include_usage")
STRATEGY_PARAMS = (
    "search_depth", "topic", "time_range", "start_date", "end_date",
    "include_domains", "include_domains_mode", "exact_match", "max_age_hours",
)


class TavilyError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class SearchOutcome:
    results: list = field(default_factory=list)
    live: bool = False            # True when Tavily fetched the pages live for this request
    degraded: list = field(default_factory=list)  # human-readable notes about lost strategy
    credits: int = 0


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def is_official(url: str, official_domains) -> bool:
    host = domain_of(url)
    return any(host == d or host.endswith("." + d) for d in (official_domains or ()))


def _post(path: str, payload: dict) -> requests.Response:
    s = get_settings()
    s.require_tavily()
    headers = {"Authorization": f"Bearer {s.tavily_key}", "Content-Type": "application/json"}
    try:
        return requests.post(f"{s.tavily_url}{path}", json=payload, headers=headers, timeout=s.timeout)
    except requests.RequestException as exc:
        raise TavilyError(0, f"Cannot reach Tavily ({type(exc).__name__}).") from exc


def build_search_payload(query: str, strategy: dict, max_results: int = 5) -> dict:
    payload = {
        "query": query[:400],
        "max_results": max_results,      # set explicitly: the docs disagree on the default
        "include_published_date": True,  # beta; value may be null
        "include_usage": True,
    }
    for key in STRATEGY_PARAMS:
        if key in strategy and strategy[key] not in (None, "", []):
            payload[key] = strategy[key]
    if "include_domains" not in payload:
        payload.pop("include_domains_mode", None)  # 400 if sent without include_domains
    return payload


def search(query: str, strategy: dict, ledger: Ledger, claim_id: str = "search") -> SearchOutcome:
    payload = build_search_payload(query, strategy)
    outcome = SearchOutcome()
    wants_live = payload.get("max_age_hours") == 0

    started = time.monotonic()
    resp = _post("/search", payload)

    if resp.status_code == 434 and "max_age_hours" in payload:
        payload.pop("max_age_hours")
        wants_live = False
        outcome.degraded.append("Live fetch refused by Tavily (HTTP 434); evidence may come from Tavily's cache.")
        resp = _post("/search", payload)

    if resp.status_code in (400, 422):
        lost = sorted(k for k in payload if k not in CORE_PARAMS)
        payload = {k: v for k, v in payload.items() if k in CORE_PARAMS}
        wants_live = False
        outcome.degraded.append("Tavily rejected advanced parameters (HTTP %d); retried without: %s." % (resp.status_code, ", ".join(lost)))
        resp = _post("/search", payload)

    if resp.status_code == 429:
        time.sleep(min(float(resp.headers.get("Retry-After", "2") or 2), 5.0))
        resp = _post("/search", payload)

    latency_ms = int((time.monotonic() - started) * 1000)
    if resp.status_code >= 400:
        ledger.add(kind="tavily", stage="search", claim_id=claim_id, latency_ms=latency_ms, credits=0, status=resp.status_code)
        raise TavilyError(resp.status_code, f"Tavily search failed (HTTP {resp.status_code}): {resp.text[:200]}")

    data = resp.json()
    outcome.live = wants_live
    outcome.credits = int(((data.get("usage") or {}).get("credits")) or 0)
    official = strategy.get("include_domains") or []
    for item in data.get("results") or []:
        url = item.get("url") or ""
        if not url:
            continue
        outcome.results.append(
            {
                "url": url,
                "title": (item.get("title") or "")[:200],
                "content": (item.get("content") or "")[:1500],
                "score": float(item.get("score") or 0),
                "published_date": normalise_date(item.get("published_date")),
                "domain": domain_of(url),
                "official": is_official(url, official),
                "live": outcome.live,
                "via": "search",
            }
        )
    ledger.add(kind="tavily", stage="search", claim_id=claim_id, latency_ms=latency_ms, credits=outcome.credits,
               status=resp.status_code, live=outcome.live, results=len(outcome.results))
    return outcome


def extract(url: str, query: str, depth: str, ledger: Ledger, claim_id: str = "-") -> str:
    """Pull the chunks of one page that matter for this claim. Returns '' on failure."""
    payload = {"urls": [url], "query": query[:400], "chunks_per_source": 3, "extract_depth": depth or "basic", "include_usage": True}
    started = time.monotonic()
    try:
        resp = _post("/extract", payload)
    except TavilyError:
        return ""
    latency_ms = int((time.monotonic() - started) * 1000)
    if resp.status_code >= 400:
        ledger.add(kind="tavily", stage="extract", claim_id=claim_id, latency_ms=latency_ms, credits=0, status=resp.status_code)
        return ""
    data = resp.json()
    credits = int(((data.get("usage") or {}).get("credits")) or 0)
    ledger.add(kind="tavily", stage="extract", claim_id=claim_id, latency_ms=latency_ms, credits=credits, status=resp.status_code)
    for item in data.get("results") or []:
        if item.get("raw_content"):
            return str(item["raw_content"])[:4000]
    return ""


def normalise_date(value) -> str | None:
    """Tavily dates arrive in several formats; keep YYYY-MM-DD or None."""
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-" and text[:4].isdigit():
        return text[:10]
    from email.utils import parsedate_to_datetime

    try:
        return parsedate_to_datetime(text).date().isoformat()
    except (TypeError, ValueError):
        return None
