"""Stage 1: find the time-sensitive claims in an answer (cheap model via Bifrost)."""

from __future__ import annotations

from . import bifrost
from .config import get_settings, load_claim_types
from .ledger import Ledger
from .textutil import find_span, parse_json_object

SYSTEM = """You extract TIME-SENSITIVE factual claims from an AI-generated answer.
A claim is time-sensitive when it can stop being true without the text changing:
model IDs, SDK or API names and versions, endpoints, prices, plans and quotas,
laws, regulations, tax rates and deadlines, company leadership and ownership.
Ignore opinions, definitions, maths, and facts that do not change over time.

Return ONLY a JSON object:
{"claims":[{"text":"<verbatim span copied from the answer, at most 200 characters>",
"type":"<one of: TYPES>",
"subject":"<the exact identifier, product, law or person the claim is about>",
"search_query":"<a short web search query that would find the current status>"}]}
Rules: copy "text" exactly from the answer. One claim per distinct fact. At most MAX claims,
most volatile first. If there are none, return {"claims":[]}."""


def extract_claims(answer: str, ledger: Ledger) -> list:
    s = get_settings()
    types = load_claim_types()
    system = SYSTEM.replace("TYPES", ", ".join(types)).replace("MAX", str(s.max_claims))
    result = bifrost.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": answer[:12000]}],
        model=s.model_extract,
        stage="extract",
        ledger=ledger,
        cache_ttl="10m",  # same answer text -> same claims; caching this stage cannot hide staleness
    )
    data = parse_json_object(result.content) or {}
    claims = []
    for raw in data.get("claims") or []:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        ctype = str(raw.get("type") or "other").strip()
        if ctype not in types:
            ctype = "other"
        claims.append(
            {
                "id": f"c{len(claims) + 1}",
                "text": text[:300],
                "span": find_span(text, answer),
                "type": ctype,
                "type_label": types[ctype]["label"],
                "subject": str(raw.get("subject") or "").strip()[:120],
                "search_query": str(raw.get("search_query") or "").strip()[:200],
            }
        )
        if len(claims) >= s.max_claims:
            break
    return claims
