"""Stage 2: gather dated evidence for one claim with a Tavily strategy chosen by claim type."""

from __future__ import annotations

from . import tavily
from .ledger import Ledger

MAX_EVIDENCE = 4


def build_query(claim: dict, ctype: dict) -> str:
    strategy = ctype.get("tavily", {})
    subject = claim.get("subject") or ""
    query = claim.get("search_query") or subject or claim["text"]
    if strategy.get("exact_match") and subject and f'"{subject}"' not in query:
        # exact_match only applies to quoted phrases
        query = f'"{subject}" ' + query.replace(subject, "").strip()
    suffix = ctype.get("query_suffix") or ""
    if suffix and suffix.split()[0].lower() not in query.lower():
        query = f"{query} {suffix}"
    return " ".join(query.split())


def gather(claim: dict, ctype: dict, ledger: Ledger) -> dict:
    """Return {'evidence': [...], 'notes': [...], 'query': str, 'live': bool}."""
    strategy = dict(ctype.get("tavily", {}))
    query = build_query(claim, ctype)
    notes: list = []

    outcome = tavily.search(query, strategy, ledger, claim["id"])
    notes += outcome.degraded

    if not outcome.results and (strategy.get("exact_match") or strategy.get("time_range")):
        relaxed = {k: v for k, v in strategy.items() if k not in ("exact_match", "time_range")}
        notes.append("No results with the strict strategy; retried without exact_match/time_range.")
        outcome = tavily.search(query.replace('"', ""), relaxed, ledger, claim["id"])
        notes += outcome.degraded

    # official sources first, then Tavily's relevance score
    ranked = sorted(outcome.results, key=lambda r: (not r["official"], -r["score"]))[:MAX_EVIDENCE]

    if strategy.get("extract") and ranked:
        top = ranked[0]
        text = tavily.extract(top["url"], claim["text"], strategy.get("extract_depth", "basic"), ledger, claim["id"])
        if text:
            # Extract returns no date: keep the published_date that Search gave us.
            top["content"] = text
            top["via"] = "search+extract"

    return {"evidence": ranked, "notes": notes, "query": query, "live": outcome.live}
