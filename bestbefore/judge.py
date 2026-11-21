"""Stage 3: a stronger model (via Bifrost) reads the evidence and answers narrow questions.

The model never sets the label. It only reports what the evidence says, with a verbatim
quote that code then checks against the source text (see textutil.quote_match).
"""

from __future__ import annotations

import re
from datetime import date

from . import bifrost
from .config import get_settings
from .ledger import Ledger
from .textutil import parse_json_object, quote_match

VERDICTS = ("supports", "contradicts", "superseded", "scheduled_end", "insufficient")
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")

SYSTEM = """You check ONE claim from an AI-generated answer against web evidence fetched today.
Use ONLY the evidence given. Do not use your own knowledge: it may be out of date.

Return ONLY a JSON object:
{"verdict":"supports|contradicts|superseded|scheduled_end|insufficient",
"end_date":"YYYY-MM-DD or null",
"replacement":"what replaces it or the current correct value, or null",
"evidence_index":<number of the evidence item you rely on, or null>,
"quote":"<verbatim excerpt of at most 30 words copied from that evidence item>",
"reasoning":"<one sentence>"}

Verdicts:
- supports: the evidence shows the claim is true today and gives no end date.
- scheduled_end: the claim is still true today but the evidence gives a FUTURE date when it stops being true (deprecation, retirement, repeal, expiry).
- superseded: the thing in the claim was retired, shut down, repealed, replaced or changed; give the date it happened in end_date if the evidence states it.
- contradicts: the evidence states a different current value (price, rate, person, date) without a clear retirement event.
- insufficient: the evidence does not settle it.
end_date must be a date stated in the evidence. Never guess a date. The quote must be copied exactly."""


def _format_evidence(evidence: list) -> str:
    blocks = []
    for i, item in enumerate(evidence, 1):
        blocks.append(
            f"[{i}] {item['title']}\nURL: {item['url']}\n"
            f"Published or updated (Tavily estimate): {item['published_date'] or 'unknown'}\n"
            f"Fetched live today: {'yes' if item['live'] else 'no'}\n"
            f"Text: {item['content']}"
        )
    return "\n\n".join(blocks)


def judge(claim: dict, evidence: list, today: date, ledger: Ledger, model: str | None = None, stage: str = "judge") -> dict:
    """Return a normalised verdict dict. Never raises on a malformed model reply."""
    if not evidence:
        return _empty("No evidence was found.")
    s = get_settings()
    user = (
        f"Today's date: {today.isoformat()}\n\nClaim: {claim['text']}\n"
        f"Subject: {claim.get('subject') or '-'}\n\nEvidence:\n{_format_evidence(evidence)}"
    )
    result = bifrost.chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        model=model or s.model_judge,
        stage=stage,
        ledger=ledger,
        claim_id=claim["id"] if stage == "judge" else f"{claim['id']}-{stage}",
        cache_ttl=None,  # never cache a freshness verdict
    )
    data = parse_json_object(result.content)
    if not data:
        out = _empty("The judge model did not return valid JSON.")
    else:
        out = normalise(data, evidence)
    out["model"] = result.model_used
    out["is_fallback"] = result.is_fallback
    return out


def normalise(data: dict, evidence: list) -> dict:
    verdict = str(data.get("verdict") or "").strip().lower()
    if verdict not in VERDICTS:
        verdict = "insufficient"

    end_date = data.get("end_date")
    end_date = end_date.strip()[:10] if isinstance(end_date, str) else None
    if end_date and not _ISO.match(end_date):
        end_date = None
    if end_date:
        try:
            date.fromisoformat(end_date)
        except ValueError:
            end_date = None

    index = data.get("evidence_index")
    try:
        index = int(index) - 1
    except (TypeError, ValueError):
        index = None
    if index is None or not 0 <= index < len(evidence):
        index = None

    quote = str(data.get("quote") or "").strip()[:400]
    match, chosen = "none", index
    if quote:
        candidates = [index] if index is not None else []
        candidates += [i for i in range(len(evidence)) if i not in candidates]
        for i in candidates:  # the model sometimes cites the wrong item number
            found = quote_match(quote, evidence[i]["content"] + " " + evidence[i]["title"])
            if found != "none":
                match, chosen = found, i
                break

    replacement = data.get("replacement")
    replacement = str(replacement).strip()[:200] if replacement not in (None, "", "null") else None
    return {
        "verdict": verdict,
        "end_date": end_date,
        "replacement": replacement,
        "evidence_index": chosen,
        "quote": quote,
        "quote_match": match,
        "reasoning": str(data.get("reasoning") or "").strip()[:400],
    }


VERDICT_CLASS = {"supports": "holds", "scheduled_end": "ends_later", "superseded": "ended", "contradicts": "ended", "insufficient": "unknown"}


def agree(a: dict, b: dict) -> bool:
    """Two independent verdicts agree when they put the claim in the same class and,
    where either states an end date, they state the same one."""
    if VERDICT_CLASS[a["verdict"]] != VERDICT_CLASS[b["verdict"]]:
        return False
    if a["end_date"] or b["end_date"]:
        return a["end_date"] == b["end_date"]
    return True


def _empty(reason: str) -> dict:
    return {
        "verdict": "insufficient", "end_date": None, "replacement": None, "evidence_index": None,
        "quote": "", "quote_match": "none", "reasoning": reason, "model": "", "is_fallback": False,
    }
