"""Orchestrates one check: claims -> evidence -> verdict -> label -> ledger."""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor

from . import bifrost, claims as claims_stage, judge as judge_stage, score, verify
from .config import get_settings, load_claim_types, today as get_today
from .ledger import Ledger

STATUS_WORDS = {
    score.FRESH: "FRESH", score.EXPIRING: "EXPIRES", score.EXPIRED: "EXPIRED", score.UNVERIFIED: "UNVERIFIED",
}


def new_ledger() -> Ledger:
    return Ledger(run_id="bb-" + uuid.uuid4().hex[:10])


def _check_claim(claim: dict, types: dict, today, ledger: Ledger) -> dict:
    ctype = types[claim["type"]]
    out = dict(claim)
    out.update(evidence=None, other_sources=[], inference=None, notes=[], error=None, strategy_why=ctype.get("why", ""))
    try:
        bundle = verify.gather(claim, ctype, ledger)
        out["notes"] = bundle["notes"]
        out["query"] = bundle["query"]
        verdict = judge_stage.judge(claim, bundle["evidence"], today, ledger)
        cross = None
        s = get_settings()
        if s.model_judge_2 and bundle["evidence"]:
            second = judge_stage.judge(claim, bundle["evidence"], today, ledger, model=s.model_judge_2, stage="judge2")
            agreed = judge_stage.agree(verdict, second)
            cross = {"model": second.get("model", ""), "verdict": second["verdict"], "end_date": second["end_date"],
                     "agrees": agreed, "is_fallback": second.get("is_fallback", False)}
    except Exception as exc:  # one failing claim must not sink the whole report
        out["error"] = str(exc)[:300]
        out.update(score.label(verdict="insufficient", end_date=None, quote_match="none", evidence_date=None,
                               live=False, official=False, volatility_days=ctype["volatility_days"], today=today))
        out["reason"] = "Check failed: " + out["error"]
        return out

    chosen = bundle["evidence"][verdict["evidence_index"]] if verdict["evidence_index"] is not None else None
    disagreement = cross is not None and not cross["agrees"]
    out.update(
        score.label(
            verdict="insufficient" if disagreement else verdict["verdict"],
            end_date=verdict["end_date"],
            quote_match=verdict["quote_match"],
            evidence_date=chosen["published_date"] if chosen else None,
            live=bool(chosen and chosen["live"]),
            official=bool(chosen and chosen["official"]),
            volatility_days=ctype["volatility_days"],
            today=today,
        )
    )
    if disagreement:
        out["reason"] = (f"Two models disagree ({verdict['verdict']} vs {cross['verdict']}), so no label is given. "
                         "Read the evidence and decide.")
    out["replacement"] = verdict["replacement"]
    # Evidence (what the page says) is kept apart from inference (what the model concluded).
    if chosen:
        out["evidence"] = {
            "url": chosen["url"], "title": chosen["title"], "domain": chosen["domain"],
            "published_date": chosen["published_date"], "official": chosen["official"], "live": chosen["live"],
            "via": chosen["via"], "quote": verdict["quote"], "quote_match": verdict["quote_match"],
        }
    out["other_sources"] = [
        {"url": e["url"], "title": e["title"], "domain": e["domain"], "published_date": e["published_date"]}
        for e in bundle["evidence"] if e is not chosen
    ]
    out["inference"] = {
        "verdict": verdict["verdict"], "reasoning": verdict["reasoning"],
        "model": verdict.get("model", ""), "is_fallback": verdict.get("is_fallback", False),
        "cross_check": cross,
    }
    return out


def check_answer(answer: str, ledger: Ledger | None = None) -> dict:
    """Label every time-sensitive claim in `answer`. Returns a JSON-serialisable report."""
    s = get_settings()
    s.require_models()
    s.require_tavily()
    ledger = ledger or new_ledger()
    today = get_today()
    types = load_claim_types()

    found = claims_stage.extract_claims(answer, ledger)
    with ThreadPoolExecutor(max_workers=max(1, s.workers)) as pool:
        checked = list(pool.map(lambda c: _check_claim(c, types, today, ledger), found))

    ledger.enrich_from_bifrost(s.bifrost_url, s.virtual_key)
    per_claim = ledger.by_claim()
    for c in checked:
        c["ledger"] = per_claim.get(c["id"], {})

    counts = {k: sum(1 for c in checked if c["status"] == k) for k in (score.FRESH, score.EXPIRING, score.EXPIRED, score.UNVERIFIED)}
    return {
        "run_id": ledger.run_id,
        "checked_on": today.isoformat(),
        "answer": answer,
        "stamp": score.answer_stamp(checked),
        "counts": counts,
        "claims": checked,
        "ledger": {"totals": ledger.totals(), "entries": ledger.entries, "notes": ledger.notes},
    }


def label_line(c: dict) -> str:
    """One plain-text line per claim, used when labels are appended to a proxied answer."""
    status = c["status"]
    if status == score.EXPIRED:
        head = f"[EXPIRED{' ' + c['expired_on'] if c.get('expired_on') else ''}]"
    elif status == score.EXPIRING:
        when = c.get("best_before")
        head = f"[BEST BEFORE {when}, {c['days_left']} days left]" if when else "[EXPIRING, no date stated]"
    elif status == score.FRESH:
        head = f"[FRESH, re-check by {c['best_before']} (estimate)]"
    else:
        head = "[UNVERIFIED]"
    line = f'{head} "{c["text"]}"'
    if c.get("replacement") and status in (score.EXPIRED, score.EXPIRING):
        line += f" -> now: {c['replacement']}"
    if c.get("evidence"):
        line += f" (source: {c['evidence']['url']})"
    return line


def labels_text(report: dict) -> str:
    if not report["claims"]:
        return ""
    lines = [f"--- BestBefore freshness labels (checked {report['checked_on']}) ---"]
    lines += [label_line(c) for c in report["claims"]]
    return "\n".join(lines)


REWRITE_SYSTEM = """You correct an AI-generated answer using verified findings.
Change ONLY the claims listed as expired or expiring, using the replacement or current value given.
For an expiring claim, keep it and add the end date in parentheses.
Keep everything else word for word. Do not add new facts. Return only the corrected answer."""


def rewrite_answer(report: dict, ledger: Ledger) -> str | None:
    """Optional: ask the judge model (via Bifrost) to rewrite stale claims. Returns None if nothing to fix."""
    fixes = [c for c in report["claims"] if c["status"] in (score.EXPIRED, score.EXPIRING) and c.get("evidence")]
    if not fixes:
        return None
    findings = "\n".join(
        f'- "{c["text"]}": {c["status"]}'
        + (f', ended {c["expired_on"]}' if c.get("expired_on") else "")
        + (f', ends {c["best_before"]}' if c["status"] == score.EXPIRING and c.get("best_before") else "")
        + (f', replacement or current value: {c["replacement"]}' if c.get("replacement") else "")
        + f' (source says: "{c["evidence"]["quote"]}")'
        for c in fixes
    )
    result = bifrost.chat(
        [{"role": "system", "content": REWRITE_SYSTEM},
         {"role": "user", "content": f"Answer:\n{report['answer']}\n\nVerified findings:\n{findings}"}],
        model=get_settings().model_judge, stage="rewrite", ledger=ledger, cache_ttl=None,
    )
    return result.content.strip() or None
