import json

from bestbefore import evaluate, pipeline, scan, score


def test_regulation_claims_use_the_law_strategy(world):
    report = pipeline.check_answer("Under the EU AI Act, Annex III start to apply on 2 August 2026. "
                                   "Transparency duties for chatbots under Article 50 apply from 2 August 2026.")
    by = {c["subject"]: c for c in report["claims"]}
    assert by["EU AI Act Annex III"]["status"] == score.EXPIRED and by["EU AI Act Annex III"]["replacement"] == "2 December 2027"
    art50 = by["EU AI Act Article 50"]
    assert art50["status"] == score.FRESH and art50["best_before_kind"] == "estimate"
    assert art50["best_before"] == "2027-02-13"  # 90-day volatility for laws, marked as an estimate
    searches = [p for s, path, p, _ in world.calls if s == "tavily" and path == "/search"]
    assert all(p.get("time_range") == "year" and "max_age_hours" not in p for p in searches)


def test_scan_fails_the_build_on_expired_model_ids(world, tmp_path, capsys):
    (tmp_path / "app.py").write_text('client.chat.completions.create(model="gpt-4-turbo", messages=m)\n')
    (tmp_path / "notes.txt").write_text('model="gpt-4-turbo"\n')         # not a scanned extension
    (tmp_path / "util.py").write_text("def add(a, b):\n    return a + b\n")  # no model hints -> not sent
    out_json = tmp_path / "r.json"
    code = scan.main([str(tmp_path), "--json", str(out_json)])
    printed = capsys.readouterr().out
    assert code == 1 and "1 file(s) checked" in printed and "[EXPIRED" in printed
    assert json.loads(out_json.read_text())[0]["file"].endswith("app.py")


def test_scan_can_also_fail_on_expiring(world, tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text('model = "gpt-5"\n')
    world.judge_override = {"verdict": "scheduled_end", "end_date": "2026-12-11", "replacement": "gpt-5.1",
                            "evidence_index": 1, "quote": "gpt-5: shutdown date 2026-12-11", "reasoning": "future"}
    assert scan.main([str(tmp_path)]) == 0
    assert scan.main([str(tmp_path), "--fail-on", "expiring"]) == 1


def test_evaluate_skips_unverified_rows_and_reports_metrics(world, tmp_path, capsys):
    rows = [
        {"answer": "x", "claim": "gpt-4-turbo", "expected": "expired", "verified_on": "2026-11-15"},
        {"answer": "x", "claim": "gpt-5", "expected": "expiring", "verified_on": "2026-11-15"},
        {"answer": "x", "claim": "dall-e-3", "expected": "fresh", "verified_on": "2026-11-15"},   # deliberately wrong label
        {"answer": "x", "claim": "Assistants API", "expected": "expired", "verified_on": None},  # skipped
    ]
    path = tmp_path / "claims.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert evaluate.main([str(path), "--json", str(tmp_path / "res.json")]) == 0
    res = json.loads((tmp_path / "res.json").read_text())
    assert res["n"] == 3 and res["skipped_unverified_rows"] == 1
    assert res["accuracy_when_labelled"] == 2 / 3
    assert res["per_label"]["expired"]["precision"] == 0.5  # dall-e-3 predicted expired, expected fresh


def test_metrics_counts_abstentions_separately():
    m = evaluate.metrics([("expired", "expired"), ("fresh", "unverified"), ("expiring", "missed")])
    assert m["abstention_rate"] == 2 / 3 and m["accuracy_when_labelled"] == 1.0 and m["accuracy_overall"] == 1 / 3


def test_cross_model_agreement_keeps_labels(world, monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_2", "other/judge")
    from .conftest import ANSWER
    report = pipeline.check_answer(ANSWER)
    judge_models = [b["model"] for s, _, b, h in world.calls if s == "bifrost" and h["x-bf-lh-stage"] in ("judge", "judge2")]
    assert judge_models.count("strong/large") == 4 and judge_models.count("other/judge") == 4
    assert report["counts"]["expired"] == 3
    assert all(c["inference"]["cross_check"]["agrees"] for c in report["claims"])


def test_cross_model_disagreement_withholds_the_label(world, monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_2", "other/judge")
    world.judge_by_model["other/judge"] = {"verdict": "supports", "end_date": None, "replacement": None, "evidence_index": 1,
                                           "quote": "gpt-4-turbo: shutdown date 2026-10-23", "reasoning": "disagrees"}
    from .conftest import ANSWER
    report = pipeline.check_answer(ANSWER)
    assert all(c["status"] == score.UNVERIFIED for c in report["claims"])
    assert "disagree" in report["claims"][0]["reason"]
