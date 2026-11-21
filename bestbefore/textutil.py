"""Small text helpers: tolerant JSON parsing and quote verification."""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
# A token is a run of letters/digits, optionally joined by . _ - / (so "gpt-5.1" is ONE token
# and never matches "gpt-5").
_TOKEN = re.compile(r"[a-z0-9]+(?:[._\-/][a-z0-9]+)*")

MAX_QUOTE_TOKENS = 40   # the prompt asks for <= 30 words; longer quotes are rejected
APPROX_SHARE = 0.85     # share of quote tokens that must appear, in order
APPROX_WINDOW = 1.5     # ... inside a source window at most 1.5x the quote length


def parse_json_object(text: str):
    """Return the first JSON object found in an LLM reply, or None."""
    if not text:
        return None
    cleaned = _FENCE.sub("", text.strip())
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    start = cleaned.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(cleaned[start : i + 1])
                        if isinstance(value, dict):
                            return value
                    except ValueError:
                        pass
                    break
        start = cleaned.find("{", start + 1)
    return None


def tokens(text: str) -> list:
    return _TOKEN.findall((text or "").lower())


def quote_match(quote: str, source: str) -> str:
    """How well does the model's quote appear in the source text?

    'exact'  - the quote's tokens appear contiguously in the source
    'approx' - >= 85% of the quote's tokens appear in order inside a source window
               no longer than 1.5x the quote (tolerates small paraphrase / dropped words)
    'none'   - not found, too short (< 4 tokens) or too long (> 40 tokens)
    Tokens are whole words, so 'gpt-5' never matches inside 'gpt-5.1'.
    """
    q, s = tokens(quote), tokens(source)
    if len(q) < 4 or len(q) > MAX_QUOTE_TOKENS or not s:
        return "none"
    n = len(q)
    for i in range(len(s) - n + 1):
        if s[i : i + n] == q:
            return "exact"
    need = APPROX_SHARE * n
    window = int(APPROX_WINDOW * n) + 1
    for start in range(len(s)):
        if s[start] != q[0] and s[start] not in q[:3]:
            continue
        found, j = 0, start
        end = min(len(s), start + window)
        for word in q:
            k = j
            while k < end and s[k] != word:
                k += 1
            if k < end:
                found += 1
                j = k + 1
        if found >= need:
            return "approx"
    return "none"


def find_span(needle: str, haystack: str):
    """Locate a claim's text inside the answer. Returns [start, end] or None."""
    if not needle:
        return None
    index = haystack.find(needle)
    if index == -1:
        index = haystack.lower().find(needle.lower())
    return [index, index + len(needle)] if index != -1 else None
