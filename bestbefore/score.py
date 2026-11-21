"""Stage 4: the label is decided by rules in code, not by a model.

Four labels, each with its own score band (0-100, higher = fresher):
  fresh      80-100  dated or live evidence supports the claim
  expiring   40-79   still true, the source states a future end date ("best before")
  expired     0-39   the source says it was retired, replaced or changed
  unverified  none   no usable evidence, conflicting signals, unverifiable quote,
                     or evidence too weak (strength < MIN_STRENGTH = 70: an exact quote from an
                     undated third-party page scores 67 and is NOT enough)

Two numbers are shown per claim and never mixed:
  * freshness score - WHERE the claim sits in time (from dates stated in the evidence)
  * evidence strength - HOW MUCH we trust the evidence (quote match + recency + source)
Evidence strength gates every label: below MIN_STRENGTH the claim is unverified.

Within a band the score is monotonic: closer to its end date -> lower; longer since it
ended -> lower. When the source gives no date, the claim gets the conservative edge of
its band (expired: 0, expiring: 40), so an undated case never outranks a dated one.
"""

from __future__ import annotations

from datetime import date, timedelta

FRESH, EXPIRING, EXPIRED, UNVERIFIED = "fresh", "expiring", "expired", "unverified"
MIN_STRENGTH = 70
HORIZON_DAYS = 180   # expiring: an end date this far away or more scores 79
DECAY_DAYS = 365     # expired: a year or more since it ended scores 0


def evidence_strength(quote_match: str, evidence_date: date | None, live: bool, official: bool, today: date) -> dict:
    """0-100 from three visible components."""
    agreement = {"exact": 60, "approx": 45}.get(quote_match, 0)
    if live:
        recency = 25
    elif evidence_date is None:
        recency = 0
    else:
        age = (today - evidence_date).days
        recency = 25 if age <= 30 else 20 if age <= 90 else 12 if age <= 365 else 5
    source = 15 if official else 7
    return {"agreement": agreement, "recency": recency, "source": source, "total": agreement + recency + source}


def expiring_score(days_left: int | None) -> int:
    if days_left is None:
        return 40
    return 40 + round(min(max(days_left, 0), HORIZON_DAYS) / HORIZON_DAYS * 39)


def expired_score(days_ago: int | None) -> int:
    if days_ago is None:
        return 0
    return 39 - round(min(max(days_ago, 0), DECAY_DAYS) / DECAY_DAYS * 39)


def fresh_score(strength_total: int) -> int:
    return 80 + round(strength_total / 100 * 20)


def label(
    *,
    verdict: str,
    end_date: str | None,
    quote_match: str,
    evidence_date: str | None,
    live: bool,
    official: bool,
    volatility_days: int,
    today: date,
) -> dict:
    ev_date = date.fromisoformat(evidence_date) if evidence_date else None
    end = date.fromisoformat(end_date) if end_date else None
    strength = evidence_strength(quote_match, ev_date, live, official, today)
    out = {
        "status": UNVERIFIED, "score": None, "evidence_strength": strength,
        "best_before": None, "best_before_kind": None, "days_left": None, "expired_on": None, "reason": "",
    }

    if verdict == "insufficient":
        out["reason"] = "The evidence does not settle this claim."
        return out
    if quote_match == "none":
        out["reason"] = "The model's quote was not found in the source text, so its verdict is not trusted."
        return out
    if strength["total"] < MIN_STRENGTH:
        out["reason"] = f"Evidence too weak to label (strength {strength['total']} < {MIN_STRENGTH})."
        return out

    if verdict in ("superseded", "contradicts", "scheduled_end") or (verdict == "supports" and end):
        if verdict == "supports" and end and end <= today:
            out["reason"] = "Conflicting signals: the model says the claim holds but cites an end date in the past."
            return out
        if end and end > today:
            days_left = (end - today).days
            out.update(status=EXPIRING, best_before=end.isoformat(), best_before_kind="source", days_left=days_left,
                       score=expiring_score(days_left),
                       reason=f"Still true today, but the source gives an end date {days_left} days from now.")
            return out
        if verdict == "scheduled_end":
            out.update(status=EXPIRING, score=expiring_score(None),
                       reason="The source announces an end but states no date; scored at the band's conservative edge.")
            return out
        if end:
            days_ago = (today - end).days
            out.update(status=EXPIRED, expired_on=end.isoformat(), score=expired_score(days_ago),
                       reason=f"The source says this stopped being true {days_ago} days ago.")
        else:
            out.update(status=EXPIRED, score=expired_score(None),
                       reason="The source states a different current value but no change date; scored at the band's conservative edge.")
        return out

    # verdict == "supports" with no end date
    if not live and ev_date is None:
        out["reason"] = "The supporting evidence has no date and was not fetched live."
        return out
    recheck = today + timedelta(days=volatility_days)
    out.update(status=FRESH, best_before=recheck.isoformat(), best_before_kind="estimate",
               days_left=volatility_days, score=fresh_score(strength["total"]),
               reason=f"Supported by current evidence. Re-check date is an estimate for this kind of fact ({volatility_days} days).")
    return out


def answer_stamp(claims: list) -> dict:
    """One stamp for the whole answer: the worst status and the earliest date."""
    statuses = [c["status"] for c in claims]
    if not claims:
        return {"status": "none", "text": "No time-sensitive claims found", "date": None}
    if EXPIRED in statuses:
        dates = sorted(c["expired_on"] for c in claims if c["status"] == EXPIRED and c.get("expired_on"))
        return {"status": EXPIRED, "text": "Expired", "date": dates[0] if dates else None}
    if EXPIRING in statuses:
        dates = sorted(c["best_before"] for c in claims if c["status"] == EXPIRING and c.get("best_before"))
        return {"status": EXPIRING, "text": "Best before", "date": dates[0] if dates else None}
    if FRESH in statuses:
        dates = sorted(c["best_before"] for c in claims if c["status"] == FRESH and c.get("best_before"))
        return {"status": FRESH, "text": "Fresh, re-check by (estimate)", "date": dates[0] if dates else None}
    return {"status": UNVERIFIED, "text": "Unverified", "date": None}
