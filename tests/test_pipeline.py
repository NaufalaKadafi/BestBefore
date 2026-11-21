from bestbefore import pipeline, score
from bestbefore.server import create_app

from .conftest import ANSWER


def by_subject(report):
    return {c["subject"]: c for c in report["claims"]}


def test_end_to_end_labels(world):
    report = pipeline.check_answer(ANSWER)
    claims = by_subject(report)
    assert claims["Assistants API"]["status"] == score.EXPIRED and claims["Assistants API"]["expired_on"] == "2026-08-26"
    assert claims["gpt-4-turbo"]["status"] == score.EXPIRED
    assert claims["dall-e-3"]["status"] == score.EXPIRED
    assert claims["gpt-5"]["status"] == score.EXPIRING and claims["gpt-5"]["days_left"] == 26
    assert report["counts"] == {"fresh": 0, "expiring": 1, "expired": 3, "unverified": 0}
    assert [claims[s]["score"] for s in ("Assistants API", "gpt-4-turbo", "dall-e-3", "gpt-5")] == [30, 37, 19, 46]
    assert report["stamp"] == {"status": "expired", "text": "Expired", "date": "2026-05-12"}
    # evidence is kept apart from inference, and comes from the official domain
    ev = claims["gpt-4-turbo"]["evidence"]
    assert ev["official"] and ev["live"] and ev["quote_match"] == "exact" and ev["published_date"] == "2026-11-10"
    assert claims["gpt-4-turbo"]["inference"]["model"] == "strong/large"
    assert all(c["span"] for c in report["claims"])


def test_every_llm_call_goes_through_bifrost_with_tags(world):
    report = pipeline.check_answer(ANSWER)
    llm = [c for c in world.calls if c[0] == "bifrost"]
    assert len(llm) == 5  # 1 extract + 4 judge
    for _, _, body, headers in llm:
        assert headers["x-bf-vk"] == "sk-bf-test"
        assert headers["x-bf-lh-run-id"] == report["run_id"]
        assert headers["x-bf-lh-stage"] in ("extract", "judge")
        assert body["fallbacks"] == ["backup/medium"]
    stages = {h["x-bf-lh-stage"]: (b["model"], h) for _, _, b, h in llm}
    assert stages["extract"][0] == "cheap/small" and stages["judge"][0] == "strong/large"
    assert stages["judge"][1].get("x-bf-cache-no-store") == "true"  # verdicts are never cached
    assert "x-bf-cache-key" in stages["extract"][1]


def test_tavily_strategy_for_model_claims(world):
    pipeline.check_answer(ANSWER)
    searches = [p for s, path, p, _ in world.calls if s == "tavily" and path == "/search"]
    assert len(searches) == 4
    for p in searches:
        assert p["max_age_hours"] == 0 and p["exact_match"] is True and p["include_domains_mode"] == "prefer"
        assert '"' in p["query"]  # exact_match needs a quoted phrase
    assert sum(1 for s, path, _, _ in world.calls if path == "/extract") == 4


def test_ledger_costs_come_from_bifrost_logs(world):
    report = pipeline.check_answer(ANSWER)
    totals = report["ledger"]["totals"]
    assert totals["llm_calls"] == 5 and totals["tavily_calls"] == 8 and totals["tavily_credits"] == 12
    assert totals["cost_usd"] == 0.002  # 5 calls x 0.0004 read back from /api/logs
    assert report["claims"][0]["ledger"]["cost_usd"] == 0.0004
    assert report["ledger"]["notes"] == []


def test_fallback_is_visible(world):
    world.fallback_for = {"c2"}
    report = pipeline.check_answer(ANSWER)
    c2 = report["claims"][1]
    assert c2["inference"]["is_fallback"] and c2["inference"]["model"] == "backup/medium"
    assert report["ledger"]["totals"]["llm_fallback_calls"] == 1


def test_tavily_434_falls_back_and_marks_evidence_cached(world):
    world.tavily_status = {"max_age_hours": 434}
    report = pipeline.check_answer(ANSWER)
    c = report["claims"][0]
    assert c["evidence"]["live"] is False
    assert any("434" in n for n in c["notes"])
    assert c["status"] == score.EXPIRED  # dated evidence still supports the verdict


def test_tavily_rejecting_new_params_degrades_to_core(world):
    world.tavily_status = {"exact_match": 400}
    report = pipeline.check_answer(ANSWER)
    assert any("advanced parameters" in n for n in report["claims"][0]["notes"])
    assert report["claims"][0]["status"] == score.EXPIRED


def test_invented_quote_is_not_trusted(world):
    world.judge_override = {"verdict": "supports", "end_date": None, "replacement": None, "evidence_index": 1,
                            "quote": "this model is supported forever and will never be retired", "reasoning": "made up"}
    report = pipeline.check_answer(ANSWER)
    assert all(c["status"] == score.UNVERIFIED for c in report["claims"])
    assert "not found in the source" in report["claims"][0]["reason"]


def test_malformed_judge_reply_is_unverified(world):
    world.judge_override = {"verdict": "banana"}
    report = pipeline.check_answer(ANSWER)
    assert all(c["status"] == score.UNVERIFIED for c in report["claims"])


def test_api_check_and_rewrite(world):
    client = create_app().test_client()
    assert client.post("/api/check", json={}).status_code == 400
    resp = client.post("/api/check", json={"answer": ANSWER, "rewrite": True})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["rewritten"] == "REWRITTEN ANSWER"
    assert data["ledger"]["totals"]["llm_calls"] == 6


def test_proxy_appends_labels_and_structured_report(world):
    client = create_app().test_client()
    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "which model?"}]})
    assert resp.status_code == 200
    data = resp.get_json()
    content = data["choices"][0]["message"]["content"]
    assert content.startswith('Use model="gpt-4-turbo" for this.')
    assert "BestBefore freshness labels (checked 2026-11-15)" in content and "[EXPIRED 2026-10-23]" in content
    assert data["bestbefore"]["counts"]["expired"] >= 1
    first = world.calls[0]
    assert first[3]["x-bf-lh-stage"] == "answer" and first[2]["stream"] is False


def test_proxy_off_mode_passes_through(world):
    client = create_app().test_client()
    resp = client.post("/v1/chat/completions", headers={"x-bb-mode": "off"}, json={"messages": [{"role": "user", "content": "hi"}]})
    assert "bestbefore" not in resp.get_json() and len(world.calls) == 1


def test_missing_models_is_a_clear_config_error(world, monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE", "")
    resp = create_app().test_client().post("/api/check", json={"answer": ANSWER})
    assert resp.status_code == 500 and "MODEL_JUDGE" in resp.get_json()["error"]["message"]


def test_static_pages(world):
    client = create_app().test_client()
    assert b"BEST" in client.get("/").data
    assert len(client.get("/api/demo").get_json()) == 4
    assert client.get("/api/health").get_json()["tavily_key"] is True
