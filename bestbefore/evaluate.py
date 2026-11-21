"""Measure label accuracy on a hand-verified set of claims.

    python -m bestbefore.evaluate eval/claims.jsonl [--json eval/results.json]

Each line of the JSONL file:
  {"answer": "...text containing the claim...", "claim": "substring of the claim",
   "expected": "fresh|expiring|expired", "source": "https://...", "verified_on": "YYYY-MM-DD", "note": "..."}

Rows without "verified_on" are SKIPPED: ground truth must be checked by a person against
the source before it counts. "unverified" predictions are abstentions and are reported
separately from wrong labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import pipeline

LABELS = ("fresh", "expiring", "expired")


def _match(claims: list, needle: str):
    n = needle.lower()
    for c in claims:
        t = c["text"].lower()
        if n in t or t in n:
            return c
    return None


def metrics(pairs: list) -> dict:
    """pairs: list of (expected, predicted); predicted may be 'unverified' or 'missed'."""
    total = len(pairs)
    abstain = sum(1 for _, p in pairs if p in ("unverified", "missed"))
    answered = [(e, p) for e, p in pairs if p not in ("unverified", "missed")]
    correct = sum(1 for e, p in answered if e == p)
    per = {}
    for lab in LABELS:
        tp = sum(1 for e, p in answered if e == lab and p == lab)
        fp = sum(1 for e, p in answered if e != lab and p == lab)
        fn = sum(1 for e, p in pairs if e == lab and p != lab)
        per[lab] = {
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "support": sum(1 for e, _ in pairs if e == lab),
        }
    confusion = {e: {p: sum(1 for x, y in pairs if x == e and y == p) for p in (*LABELS, "unverified", "missed")} for e in LABELS}
    return {
        "n": total,
        "abstention_rate": abstain / total if total else None,
        "accuracy_when_labelled": correct / len(answered) if answered else None,
        "accuracy_overall": correct / total if total else None,
        "per_label": per,
        "confusion": confusion,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate BestBefore labels on verified claims.")
    ap.add_argument("path")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args(argv)

    rows, skipped = [], 0
    for line in Path(args.path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("verified_on") or row.get("expected") not in LABELS:
            skipped += 1
            continue
        rows.append(row)
    if not rows:
        print(f"No verified rows (skipped {skipped}). Fill 'verified_on' after checking each source by hand.")
        return 2

    pairs, details = [], []
    for row in rows:
        report = pipeline.check_answer(row["answer"])
        found = _match(report["claims"], row["claim"])
        predicted = found["status"] if found else "missed"
        pairs.append((row["expected"], predicted))
        details.append({"claim": row["claim"], "expected": row["expected"], "predicted": predicted,
                        "score": found.get("score") if found else None, "run_id": report["run_id"]})
        print(f"{'OK ' if predicted == row['expected'] else '-- '} expected {row['expected']:9s} got {predicted:10s} {row['claim'][:70]}")

    result = metrics(pairs)
    result["skipped_unverified_rows"] = skipped
    result["details"] = details
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
