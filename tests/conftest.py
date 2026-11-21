"""Offline test doubles for Bifrost and Tavily.

These fakes exist ONLY in the test suite. The application has no mock mode: a real run
always calls the real gateway and the real search API. The page texts below are test
fixtures, not claims about the real world.
"""

import json

import pytest
import requests

FIXTURE_PAGE = (
    "Deprecations. Assistants API: shutdown date 2026-08-26, use the Responses API instead. "
    "gpt-4-turbo: shutdown date 2026-10-23, recommended replacement gpt-5.1. "
    "dall-e-3: shutdown date 2026-05-12, recommended replacement gpt-image-1. "
    "gpt-5: shutdown date 2026-12-11, recommended replacement gpt-5.1. "
    "text-embedding-3-small is generally available with no shutdown date announced."
)

CLAIMS = [
    {"text": "create the assistant with the Assistants API", "type": "model_or_api", "subject": "Assistants API", "search_query": "OpenAI Assistants API deprecation"},
    {"text": 'model="gpt-4-turbo"', "type": "model_or_api", "subject": "gpt-4-turbo", "search_query": "gpt-4-turbo deprecation"},
    {"text": "call the images endpoint with dall-e-3", "type": "model_or_api", "subject": "dall-e-3", "search_query": "dall-e-3 deprecation"},
    {"text": 'model="gpt-5"', "type": "model_or_api", "subject": "gpt-5", "search_query": "gpt-5 deprecation"},
]

LAW_PAGE = (
    "Digital omnibus in force. High-risk obligations for Annex III systems now apply from 2 December 2027 instead of "
    "2 August 2026. Article 50 transparency obligations for chatbots apply from 2 August 2026."
)

LAW_CLAIMS = [
    {"text": "Annex III start to apply on 2 August 2026", "type": "law_regulation", "subject": "EU AI Act Annex III", "search_query": "EU AI Act Annex III high-risk application date"},
    {"text": "Transparency duties for chatbots under Article 50 apply from 2 August 2026", "type": "law_regulation", "subject": "EU AI Act Article 50", "search_query": "EU AI Act Article 50 transparency date"},
]

VERDICTS = {
    "EU AI Act Annex III": {"verdict": "superseded", "end_date": None, "replacement": "2 December 2027", "evidence_index": 1,
                            "quote": "High-risk obligations for Annex III systems now apply from 2 December 2027", "reasoning": "Moved."},
    "EU AI Act Article 50": {"verdict": "supports", "end_date": None, "replacement": None, "evidence_index": 1,
                             "quote": "Article 50 transparency obligations for chatbots apply from 2 August 2026", "reasoning": "Still applies."},
    "Assistants API": {"verdict": "superseded", "end_date": "2026-08-26", "replacement": "Responses API", "evidence_index": 1,
                       "quote": "Assistants API: shutdown date 2026-08-26", "reasoning": "The page lists a past shutdown date."},
    "gpt-4-turbo": {"verdict": "superseded", "end_date": "2026-10-23", "replacement": "gpt-5.1", "evidence_index": 1,
                    "quote": "gpt-4-turbo: shutdown date 2026-10-23", "reasoning": "The page lists a past shutdown date."},
    "dall-e-3": {"verdict": "superseded", "end_date": "2026-05-12", "replacement": "gpt-image-1", "evidence_index": 1,
                 "quote": "dall-e-3: shutdown date 2026-05-12", "reasoning": "The page lists a past shutdown date."},
    "gpt-5": {"verdict": "scheduled_end", "end_date": "2026-12-11", "replacement": "gpt-5.1", "evidence_index": 1,
              "quote": "gpt-5: shutdown date 2026-12-11", "reasoning": "The page lists a future shutdown date."},
}


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeWorld:
    """Routes requests.post / requests.get to canned Bifrost and Tavily behaviour."""

    def __init__(self):
        self.calls = []            # (service, path, payload, headers)
        self.tavily_status = {}    # e.g. {"max_age_hours": 434} -> fail while that key is in the payload
        self.judge_override = None  # dict returned by the judge for every claim
        self.judge_by_model = {}     # model -> dict, overrides judge_override for that model
        self.fallback_for = set()   # claim ids answered by a fallback model
        self.logs = []

    # -- Bifrost ---------------------------------------------------------
    def _completion(self, content, body, headers):
        claim_id = headers.get("x-bf-lh-claim-id", "-").split(":")[-1]
        is_fallback = claim_id in self.fallback_for
        model = body["fallbacks"][0] if is_fallback and body.get("fallbacks") else body["model"]
        self.logs.append({"cost": 0.0004, "metadata": {"run-id": headers["x-bf-lh-run-id"], "stage": headers["x-bf-lh-stage"], "claim-id": headers["x-bf-lh-claim-id"]}})
        return FakeResponse(200, {
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 40},
            "extra_fields": {"routing_info": {"provider": model.split("/")[0], "model": model, "is_fallback": is_fallback}},
        })

    def bifrost(self, body, headers):
        stage = headers.get("x-bf-lh-stage")
        if stage == "extract":
            text = body["messages"][-1]["content"]
            claims = LAW_CLAIMS if "AI Act" in text else CLAIMS
            return self._completion("```json\n" + json.dumps({"claims": claims}) + "\n```", body, headers)
        if stage in ("judge", "judge2"):
            user = body["messages"][-1]["content"]
            if body["model"] in self.judge_by_model:
                return self._completion(json.dumps(self.judge_by_model[body["model"]]), body, headers)
            if self.judge_override is not None:
                return self._completion(json.dumps(self.judge_override), body, headers)
            subject = user.split("Subject: ")[1].split("\n")[0]
            return self._completion(json.dumps(VERDICTS[subject]), body, headers)
        if stage == "rewrite":
            return self._completion("REWRITTEN ANSWER", body, headers)
        return self._completion("Use model=\"gpt-4-turbo\" for this.", body, headers)  # stage == answer

    # -- Tavily ----------------------------------------------------------
    def tavily(self, path, payload):
        for key, status in self.tavily_status.items():
            if key in payload:
                return FakeResponse(status, {"detail": f"{key} not available"})
        if "AI Act" in payload.get("query", ""):
            return FakeResponse(200, {"results": [{"url": "https://www.goodwinlaw.com/alert", "title": "EU AI Act update", "content": LAW_PAGE,
                                                    "score": 0.8, "published_date": "2026-08-05"}], "usage": {"credits": 2}})
        if path == "/extract":
            return FakeResponse(200, {"results": [{"url": payload["urls"][0], "raw_content": FIXTURE_PAGE}], "usage": {"credits": 1}})
        return FakeResponse(200, {
            "results": [
                {"url": "https://developers.openai.com/api/docs/deprecations", "title": "Deprecations", "content": FIXTURE_PAGE,
                 "score": 0.91, "published_date": "2026-11-10"},
                {"url": "https://blog.example.com/model-news", "title": "Model news", "content": "Some models were retired this year.",
                 "score": 0.55, "published_date": None},
            ],
            "usage": {"credits": 2},
        })

    # -- requests entry points -----------------------------------------
    def post(self, url, json=None, headers=None, timeout=None):
        headers = headers or {}
        if "/v1/chat/completions" in url:
            self.calls.append(("bifrost", "/v1/chat/completions", json, headers))
            return self.bifrost(json, headers)
        path = "/extract" if url.endswith("/extract") else "/search"
        self.calls.append(("tavily", path, dict(json), headers))
        return self.tavily(path, json)

    def get(self, url, params=None, headers=None, timeout=None):
        if url.endswith("/api/logs"):
            return FakeResponse(200, {"logs": self.logs})
        return FakeResponse(200, {})


@pytest.fixture
def world(monkeypatch):
    fake = FakeWorld()
    monkeypatch.setattr(requests, "post", fake.post)
    monkeypatch.setattr(requests, "get", fake.get)
    monkeypatch.setenv("BIFROST_URL", "http://bifrost.test")
    monkeypatch.setenv("BIFROST_VIRTUAL_KEY", "sk-bf-test")
    monkeypatch.setenv("MODEL_EXTRACT", "cheap/small")
    monkeypatch.setenv("MODEL_JUDGE", "strong/large")
    monkeypatch.setenv("MODEL_FALLBACKS", "backup/medium")
    monkeypatch.setenv("MODEL_CHAT", "cheap/small")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setenv("BESTBEFORE_TODAY", "2026-11-15")
    monkeypatch.setenv("WORKERS", "1")
    return fake


ANSWER = (
    "First, create the assistant with the Assistants API so OpenAI stores the thread. "
    'For reasoning use model="gpt-4-turbo". To generate the banner, call the images endpoint with dall-e-3. '
    'For the final answer use model="gpt-5".'
)
