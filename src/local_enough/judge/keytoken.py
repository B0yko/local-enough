"""Deterministic key-token check: does a summary state a required fact's owner, amount or date.

Independent of the judge model; used to report how often the LLM judge agrees with this fixed rule.
"""

from __future__ import annotations

import re
from typing import Any

_MONTHS: dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_DAY_MONTH_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\b")
_MONTH_DAY_RE = re.compile(r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\b")
_SLASH_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})\b")
_ISO_DATE_RE = re.compile(r"\b\d{4}-(\d{2})-(\d{2})\b")
_AMOUNT_RE = re.compile(r"(?:[$€£]\s?)?\d[\d,]*(?:\.\d+)?\s?[kK]?\b")


def check_owner(name: str, summary: str) -> bool:
    """Case-insensitive name-token presence."""
    return name.casefold() in summary.casefold()


def _normalize_amount(text: str) -> float | None:
    cleaned = re.sub(r"[^0-9.kK]", "", text)
    if not cleaned:
        return None
    multiplier = 1.0
    if cleaned[-1] in "kK":
        multiplier = 1000.0
        cleaned = cleaned[:-1]
    if not cleaned or cleaned.count(".") > 1:
        return None
    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None


def check_amount(amount: str, summary: str) -> bool:
    """The amount's digits, stripped of currency symbols and separators, found anywhere in the summary.

    Accepts "12.5k" and "12,500" style forms on either side of the comparison.
    """
    target = _normalize_amount(amount)
    if target is None:
        return False
    for match in _AMOUNT_RE.finditer(summary):
        value = _normalize_amount(match.group(0))
        if value is not None and abs(value - target) < 0.005:
            return True
    return False


def _month_day(text: str) -> tuple[int, int] | None:
    """(month, day) parsed from a "D Month" or "Month D" phrase, or ``None`` if it names no month."""
    match = _DAY_MONTH_RE.search(text)
    if match:
        month = _MONTHS.get(match.group(2).casefold())
        day = int(match.group(1))
        if month and 1 <= day <= 31:
            return month, day
    match = _MONTH_DAY_RE.search(text)
    if match:
        month = _MONTHS.get(match.group(1).casefold())
        day = int(match.group(2))
        if month and 1 <= day <= 31:
            return month, day
    return None


def check_date(date: str, summary: str) -> bool:
    """A calendar date ("14 April") matches any common English form in the summary: ISO, "October 14",
    "Oct 14", "14/10" or a weekday plus the date. A phrase that names no month (a weekday alone, "next
    Monday", "the 20th", "the end of the month") falls back to a case-insensitive substring match.
    """
    target = _month_day(date)
    if target is None:
        return date.casefold() in summary.casefold()
    month, day = target
    day_month = any(
        _MONTHS.get(m.group(2).casefold()) == month and int(m.group(1)) == day for m in _DAY_MONTH_RE.finditer(summary)
    )
    month_day = any(
        _MONTHS.get(m.group(1).casefold()) == month and int(m.group(2)) == day for m in _MONTH_DAY_RE.finditer(summary)
    )
    slash = any(int(m.group(1)) == day and int(m.group(2)) == month for m in _SLASH_DATE_RE.finditer(summary))
    iso = any(int(m.group(1)) == month and int(m.group(2)) == day for m in _ISO_DATE_RE.finditer(summary))
    return day_month or month_day or slash or iso


def check_fact(key_tokens: dict[str, Any], summary: str) -> bool:
    """A fact passes the check when every one of its key tokens passes."""
    for kind, value in key_tokens.items():
        text = str(value)
        if kind == "owner":
            ok = check_owner(text, summary)
        elif kind == "amount":
            ok = check_amount(text, summary)
        elif kind == "date":
            ok = check_date(text, summary)
        else:
            ok = text.casefold() in summary.casefold()
        if not ok:
            return False
    return True
