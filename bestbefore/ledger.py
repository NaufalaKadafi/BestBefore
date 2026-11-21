"""Per-run ledger: every LLM call (via Bifrost) and every Tavily call, tagged by stage and claim.

Tokens and latency are measured by this app. Cost comes from Bifrost's own request
logs (GET /api/logs) on a best-effort basis: calls are tagged with x-bf-lh-* headers
so the gateway stores run id, stage and claim id as log metadata.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field

import requests


@dataclass
class Ledger:
    run_id: str
    entries: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, **entry) -> None:
        with self._lock:
            self.entries.append(entry)

    def note(self, text: str) -> None:
        with self._lock:
            if text not in self.notes:
                self.notes.append(text)

    # ---- summaries -------------------------------------------------------
    def totals(self) -> dict:
        llm = [e for e in self.entries if e["kind"] == "llm"]
        web = [e for e in self.entries if e["kind"] == "tavily"]
        costs = [e["cost_usd"] for e in llm if e.get("cost_usd") is not None]
        return {
            "llm_calls": len(llm),
            "llm_fallback_calls": sum(1 for e in llm if e.get("is_fallback")),
            "prompt_tokens": sum(e.get("prompt_tokens") or 0 for e in llm),
            "completion_tokens": sum(e.get("completion_tokens") or 0 for e in llm),
            "llm_latency_ms": sum(e.get("latency_ms") or 0 for e in llm),
            "cost_usd": round(sum(costs), 6) if costs else None,
            "tavily_calls": len(web),
            "tavily_credits": sum(e.get("credits") or 0 for e in web),
            "tavily_latency_ms": sum(e.get("latency_ms") or 0 for e in web),
        }

    def by_claim(self) -> dict:
        out: dict = {}
        for e in self.entries:
            row = out.setdefault(
                e.get("claim_id", "?"),
                {"llm_calls": 0, "tokens": 0, "latency_ms": 0, "cost_usd": None, "tavily_credits": 0, "models": []},
            )
            row["latency_ms"] += e.get("latency_ms") or 0
            if e["kind"] == "llm":
                row["llm_calls"] += 1
                row["tokens"] += (e.get("prompt_tokens") or 0) + (e.get("completion_tokens") or 0)
                if e.get("model_used") and e["model_used"] not in row["models"]:
                    row["models"].append(e["model_used"])
                if e.get("cost_usd") is not None:
                    row["cost_usd"] = round((row["cost_usd"] or 0) + e["cost_usd"], 6)
            else:
                row["tavily_credits"] += e.get("credits") or 0
        return out

    # ---- cost read-back from Bifrost ------------------------------------
    def enrich_from_bifrost(self, bifrost_url: str, virtual_key: str = "", timeout: int = 10) -> None:
        """Attach cost_usd to LLM entries using Bifrost's request logs.

        The log record shape differs between Bifrost versions, so matching is
        defensive: a log belongs to an entry when it mentions this run id, the
        stage and the claim id. If nothing matches, cost stays None and a note
        says so; it is never estimated.
        """
        headers = {"x-bf-vk": virtual_key} if virtual_key else {}
        try:
            resp = requests.get(f"{bifrost_url}/api/logs", params={"limit": 200}, headers=headers, timeout=timeout)
            resp.raise_for_status()
            logs = resp.json().get("logs") or []
        except Exception as exc:  # network, auth, shape
            self.note(f"Cost unavailable: could not read Bifrost logs ({type(exc).__name__}).")
            return

        matched = 0
        for log in logs:
            blob = json.dumps(log, default=str)
            if self.run_id not in blob:
                continue
            cost = _find_cost(log)
            if cost is None:
                continue
            for e in self.entries:
                if e["kind"] != "llm" or e.get("cost_usd") is not None:
                    continue
                # the tag is unique per run and claim: "<run_id>:<claim_id>"
                if f'{self.run_id}:{e["claim_id"]}"' in blob:
                    e["cost_usd"] = cost
                    matched += 1
                    break
        if not matched:
            self.note(
                "Cost unavailable: no Bifrost log matched this run. Check that logging is enabled "
                "and that x-bf-lh-* headers are stored as log metadata in your Bifrost version."
            )


def _find_cost(log: dict):
    """Return a numeric cost from a Bifrost log record, or None."""
    for key in ("cost", "total_cost"):
        value = log.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            for inner in ("total_cost", "total", "cost"):
                if isinstance(value.get(inner), (int, float)):
                    return float(value[inner])
            parts = [v for k, v in value.items() if k in ("input_cost", "output_cost", "additional_cost") and isinstance(v, (int, float))]
            if parts:
                return float(sum(parts))
    return None
