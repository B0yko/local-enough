"""Shared lenient parser: identical stripping and extraction rules for every model."""

from __future__ import annotations

import json
import re
from typing import Any

_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_UNCLOSED = re.compile(r"<think>.*", re.IGNORECASE | re.DOTALL)
_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)```", re.DOTALL)
LABEL_STRIP_CHARS = " \t\n\r`'\".,;:!?"


def strip_wrappers(text: str) -> str:
    """Remove ``<think>...</think>`` blocks (including an unterminated one) and code fences."""
    cleaned = _THINK_CLOSED.sub("", text)
    cleaned = _THINK_UNCLOSED.sub("", cleaned)
    fence = _FENCE.search(cleaned)
    if fence:
        cleaned = fence.group(1)
    return cleaned.strip()


def first_line_label(text: str) -> str:
    """First non-empty line, quotes/backticks/trailing punctuation stripped."""
    cleaned = strip_wrappers(text)
    for line in cleaned.splitlines():
        label = line.strip().strip(LABEL_STRIP_CHARS)
        if label:
            return label
    raise ValueError("no label found in output")


def first_json_value(text: str) -> Any:
    """The first value ``json.JSONDecoder.raw_decode`` can parse starting at a ``{`` or ``[``."""
    cleaned = strip_wrappers(text)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(cleaned, i)
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON value found in output")
