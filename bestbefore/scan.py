"""CI / pre-commit mode: scan source files for expired or expiring model IDs and API usage.

    python -m bestbefore.scan src/ [more paths] [--fail-on expired|expiring] [--json report.json]

Exit code 1 when a claim at or above --fail-on is found (default: expired), so a build
can fail before a retired model ID reaches production. Only lines near likely model/API
references are sent for checking, which keeps cost and Tavily credits down.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import pipeline, score

EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".go", ".java", ".kt", ".rb", ".php", ".cs",
              ".md", ".mdx", ".yaml", ".yml", ".toml", ".json", ".ipynb", ".env.example"}
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next", "site-packages"}
HINT = re.compile(r"(model\s*[=:]|[\"'](gpt|o\d|claude|gemini|dall-e|text-embedding|whisper|tts)[-\w.]*[\"']|"
                  r"assistants|completions|api[_-]?version|/v\d+/)", re.IGNORECASE)
CONTEXT_LINES = 2
MAX_CHARS = 12000


def iter_files(paths):
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            yield p
            continue
        for f in sorted(p.rglob("*")):
            if f.is_file() and not (set(f.parts) & SKIP_DIRS) and (f.suffix in EXTENSIONS or f.name.endswith(".env.example")):
                yield f


def snippet(path: Path):
    """Lines near model/API hints, with line numbers. None if the file has no hint."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    keep = set()
    for i, line in enumerate(lines):
        if HINT.search(line):
            keep.update(range(max(0, i - CONTEXT_LINES), min(len(lines), i + CONTEXT_LINES + 1)))
    if not keep:
        return None
    text = "\n".join(f"{path}:{i + 1}: {lines[i]}" for i in sorted(keep))
    return text[:MAX_CHARS]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--fail-on", choices=["expired", "expiring"], default="expired")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args(argv)

    failing = {score.EXPIRED} | ({score.EXPIRING} if args.fail_on == "expiring" else set())
    results, bad = [], 0
    for path in iter_files(args.paths):
        text = snippet(path)
        if not text:
            continue
        report = pipeline.check_answer(text)
        for c in report["claims"]:
            line = pipeline.label_line(c)
            print(f"{path}: {line}")
            if c["status"] in failing:
                bad += 1
        results.append({"file": str(path), "report": report})

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"BestBefore scan: {len(results)} file(s) checked, {bad} claim(s) at or above '{args.fail_on}'.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
