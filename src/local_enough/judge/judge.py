"""The summarisation judge: prompt rendering, lenient verdict parsing and the pass rule.

The judge only reports facts and unsupported claims; code decides pass/fail (:func:`decide_pass`), never the
model itself.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import jinja2

from local_enough.providers.openai_compat import Message
from local_enough.tasks import parsing

_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "judge.j2"
_MARKER = "===USER==="


def template_file() -> Path:
    """Path to the judge prompt template, for sha256 recording alongside the task templates."""
    return _TEMPLATE_PATH


def _template_parts() -> tuple[str, str]:
    raw = _TEMPLATE_PATH.read_text(encoding="utf-8")
    system_part, marker, user_part = raw.partition(_MARKER)
    if not marker:
        return "", raw.strip()
    return system_part.strip(), user_part.strip()


def render_judge_messages(transcript: str, required_facts: list[dict[str, Any]], summary: str) -> list[Message]:
    """The benchmarked judge prompt: transcript, numbered required facts and the summary under review."""
    system_tmpl, user_tmpl = _template_parts()
    env = jinja2.Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
    context = {"transcript": transcript, "required_facts": required_facts, "summary": summary}
    messages: list[Message] = []
    if system_tmpl:
        messages.append({"role": "system", "content": env.from_string(system_tmpl).render(**context)})
    messages.append({"role": "user", "content": env.from_string(user_tmpl).render(**context)})
    return messages


def parse_verdict(raw_text: str, n_facts: int) -> dict[str, Any] | None:
    """Lenient-parse a judge verdict and validate its shape; ``None`` on anything invalid.

    Valid: a JSON object with a ``facts`` list covering every index ``0 .. n_facts - 1`` exactly once (each
    entry a dict with boolean ``present``/``correct``) and an ``unsupported_claims`` list of strings.
    """
    try:
        value = parsing.first_json_value(raw_text)
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    facts = value.get("facts")
    claims = value.get("unsupported_claims")
    if not isinstance(facts, list) or not isinstance(claims, list):
        return None
    if not all(isinstance(c, str) for c in claims):
        return None

    seen: set[int] = set()
    parsed_facts: list[dict[str, Any]] = []
    for entry in facts:
        if not isinstance(entry, dict):
            return None
        index = entry.get("index")
        present = entry.get("present")
        correct = entry.get("correct")
        if not isinstance(index, int) or isinstance(index, bool):
            return None
        if not isinstance(present, bool) or not isinstance(correct, bool):
            return None
        if index in seen or not (0 <= index < n_facts):
            return None
        seen.add(index)
        parsed_facts.append({"index": index, "present": present, "correct": correct})

    if seen != set(range(n_facts)):
        return None
    parsed_facts.sort(key=lambda f: int(f["index"]))
    return {"facts": parsed_facts, "unsupported_claims": list(claims)}


def present_and_correct_count(verdict: dict[str, Any]) -> int:
    """Number of facts the verdict marks both present and correct."""
    return sum(1 for f in verdict["facts"] if f["present"] and f["correct"])


def decide_pass(verdict: dict[str, Any] | None, n_facts: int, words: int, max_words: int) -> bool:
    """Code, not the judge, decides pass: >= ceil(0.8 * n) present-and-correct, zero unsupported, within limit.

    An invalid verdict (``None``) is always a predicted fail.
    """
    if verdict is None:
        return False
    threshold = math.ceil(0.8 * n_facts)
    return (
        present_and_correct_count(verdict) >= threshold
        and len(verdict["unsupported_claims"]) == 0
        and words <= max_words
    )
