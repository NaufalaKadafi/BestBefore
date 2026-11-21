from datetime import date

from bestbefore import score
from bestbefore.tavily import build_search_payload, is_official, normalise_date
from bestbefore.textutil import find_span, parse_json_object, quote_match

TODAY = date(2026, 11, 15)


def label(**kw):
    base = dict(verdict="supports", end_date=None, quote_match="exact", evidence_date="2026-11-10",
                live=False, official=True, volatility_days=30, today=TODAY)
    base.update(kw)
    return score.label(**base)


def test_past_end_date_is_expired_with_date():
    out = label(verdict="superseded", end_date="2026-10-23")
    assert out["status"] == score.EXPIRED and out["expired_on"] == "2026-10-23"
    assert 0 <= out["score"] <= 39


def test_future_end_date_is_expiring_with_days_left():
    out = label(verdict="scheduled_end", end_date="2026-12-11")
    assert out["status"] == score.EXPIRING
    assert out["best_before"] == "2026-12-11" and out["best_before_kind"] == "source" and out["days_left"] == 26
    assert 40 <= out["score"] <= 79


def test_superseded_with_future_date_is_expiring_not_expired():
    assert label(verdict="superseded", end_date="2027-01-01")["status"] == score.EXPIRING


def test_supported_claim_is_fresh_with_estimated_recheck():
    out = label()
    assert out["status"] == score.FRESH and out["best_before"] == "2026-12-15" and out["best_before_kind"] == "estimate"
    assert 80 <= out["score"] <= 100


def test_undated_evidence_never_yields_fresh():
    assert label(evidence_date=None, live=False)["status"] == score.UNVERIFIED
    assert label(evidence_date=None, live=True)["status"] == score.FRESH  # a live fetch is dated today


def test_unverifiable_quote_blocks_any_verdict():
    for verdict in ("supports", "superseded", "contradicts", "scheduled_end"):
        assert label(verdict=verdict, end_date="2026-10-23", quote_match="none")["status"] == score.UNVERIFIED


def test_conflicting_signals_are_unverified():
    assert label(verdict="supports", end_date="2026-01-01")["status"] == score.UNVERIFIED


def test_contradiction_without_date_is_expired_at_conservative_edge():
    out = label(verdict="contradicts")
    assert out["status"] == score.EXPIRED and out["expired_on"] is None and out["score"] == 0


def test_scores_are_monotonic_and_undated_never_outranks_dated():
    expired = [score.expired_score(d) for d in (0, 23, 81, 187, 365, 900)]
    assert expired == sorted(expired, reverse=True) and expired[0] == 39 and expired[-1] == 0
    assert score.expired_score(None) <= min(expired)
    expiring = [score.expiring_score(d) for d in (1, 26, 90, 180, 400)]
    assert expiring == sorted(expiring) and expiring[-1] == 79
    assert score.expiring_score(None) <= min(expiring)
    assert label(verdict="scheduled_end", end_date=None)["score"] == 40


def test_weak_evidence_is_not_labelled():
    # an exact quote from an undated third-party page (60 + 0 + 7 = 67) is not enough on its own
    assert label(verdict="superseded", end_date="2026-10-23", evidence_date=None, official=False)["status"] == score.UNVERIFIED
    # approx quote (45) + undated (0) + third party (7) = 52 < 70
    out = label(verdict="superseded", end_date="2026-10-23", quote_match="approx", evidence_date=None, official=False)
    assert out["status"] == score.UNVERIFIED and "too weak" in out["reason"]
    # same quote from an official page fetched live is strong enough
    assert label(verdict="superseded", end_date="2026-10-23", quote_match="approx", evidence_date=None, live=True)["status"] == score.EXPIRED


def test_insufficient_is_unverified():
    assert label(verdict="insufficient")["status"] == score.UNVERIFIED


def test_answer_stamp_takes_worst_status_and_earliest_date():
    claims = [
        {"status": score.FRESH, "best_before": "2026-12-15"},
        {"status": score.EXPIRED, "expired_on": "2026-10-23"},
        {"status": score.EXPIRED, "expired_on": "2026-05-12"},
        {"status": score.EXPIRING, "best_before": "2026-12-11"},
    ]
    assert score.answer_stamp(claims) == {"status": score.EXPIRED, "text": "Expired", "date": "2026-05-12"}
    assert score.answer_stamp(claims[:1])["status"] == score.FRESH
    assert score.answer_stamp([])["status"] == "none"


def test_quote_match_levels():
    src = "The  gpt-4-turbo model:\nshutdown date 2026-10-23, recommended replacement gpt-5.1."
    assert quote_match("gpt-4-turbo model: shutdown date 2026-10-23", src) == "exact"
    assert quote_match("gpt-4-turbo model has shutdown date 2026-10-23 recommended replacement", src) == "approx"
    assert quote_match("gpt-4-turbo will remain available until 2030", src) == "none"
    # whole-token matching: gpt-5 must not match inside gpt-5.1
    assert quote_match("gpt-5: shutdown date 2026-12-11", "gpt-5.1: shutdown date 2026-12-11") == "none"
    # words scattered far apart in the source do not count as a match
    far = "gpt-4-turbo " + "filler " * 40 + "shutdown " + "filler " * 40 + "date 2026-10-23 recommended replacement"
    assert quote_match("gpt-4-turbo model: shutdown date 2026-10-23", far) == "none"
    # over-long quotes are rejected (the prompt asks for <= 30 words)
    assert quote_match(" ".join(["word"] * 45), " ".join(["word"] * 60)) == "none"
    assert quote_match("", src) == "none"


def test_parse_json_object_is_tolerant():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}
    assert parse_json_object("no json here") is None


def test_find_span():
    assert find_span("GPT-5", "use gpt-5 now") == [4, 9]
    assert find_span("missing", "use gpt-5 now") is None


def test_search_payload_follows_strategy():
    payload = build_search_payload("q", {"max_age_hours": 0, "exact_match": True, "include_domains": ["openai.com"], "include_domains_mode": "prefer", "extract": True})
    assert payload["max_age_hours"] == 0 and payload["exact_match"] is True and payload["include_domains_mode"] == "prefer"
    assert "extract" not in payload and payload["include_published_date"] is True
    # include_domains_mode without include_domains would be rejected by Tavily
    assert "include_domains_mode" not in build_search_payload("q", {"include_domains_mode": "prefer"})


def test_official_domain_and_date_helpers():
    assert is_official("https://developers.openai.com/x", ["openai.com"])
    assert not is_official("https://notopenai.com/x", ["openai.com"])
    assert normalise_date("2026-11-10T08:00:00Z") == "2026-11-10"
    assert normalise_date("Tue, 10 Nov 2026 08:00:00 GMT") == "2026-11-10"
    assert normalise_date(None) is None and normalise_date("soon") is None
