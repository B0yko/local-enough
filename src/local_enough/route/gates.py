"""Deterministic confidence gates (spec item 16): the same code for ``route.simulate`` and ``route.server``.

Every gate is a pure function of a task spec, the router's own input item (no gold), the parsed output and a
:class:`GateContext` built from the task's own dataset -- never from the model's self-reported confidence. Two
kinds of gate parameter are tuned once on ``calib`` and then frozen: the extraction grounding ratio and the
entity-matching uncertainty band (:func:`build_gate_context`); the planner stores the same numbers in the plan
so a fresh :class:`GateContext` built from the same run always reproduces them.
"""

from __future__ import annotations

import contextlib
import math
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from rapidfuzz import fuzz

from local_enough.bench.rundir import RunDir
from local_enough.providers.baselines import FuzzyMatchBaseline, RegexPiiBaseline, TfidfBaseline
from local_enough.tasks import registry
from local_enough.tasks.base import FieldSpec, Item, Parsed, TaskSpec
from local_enough.tasks.extraction import normalize_field

_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS_ABBR = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}  # fmt: skip
_MONTHS_FULL = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}  # fmt: skip
_MONTH_NAMES = sorted({*_MONTHS_ABBR, *_MONTHS_FULL}, key=len, reverse=True)
_MONTH_NAME_RE = "|".join(_MONTH_NAMES)
_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}  # fmt: skip

# ISO-3166 alpha-2 for the countries the bundled datasets use, plus other common ones. Keys are upper-cased
# name/code variants; values are the alpha-2 code. Extend as bring-your-own tasks need more.
COUNTRY_TABLE: dict[str, str] = {
    "GB": "GB", "UK": "GB", "UNITED KINGDOM": "GB", "GREAT BRITAIN": "GB",
    "US": "US", "USA": "US", "UNITED STATES": "US", "UNITED STATES OF AMERICA": "US",
    "CA": "CA", "CANADA": "CA",
    "IE": "IE", "IRELAND": "IE",
    "FR": "FR", "FRANCE": "FR",
    "DE": "DE", "GERMANY": "DE",
    "ES": "ES", "SPAIN": "ES",
    "IT": "IT", "ITALY": "IT",
    "NL": "NL", "NETHERLANDS": "NL", "THE NETHERLANDS": "NL",
    "BE": "BE", "BELGIUM": "BE",
    "PT": "PT", "PORTUGAL": "PT",
    "AT": "AT", "AUSTRIA": "AT",
    "CH": "CH", "SWITZERLAND": "CH",
    "SE": "SE", "SWEDEN": "SE",
    "NO": "NO", "NORWAY": "NO",
    "DK": "DK", "DENMARK": "DK",
    "FI": "FI", "FINLAND": "FI",
    "PL": "PL", "POLAND": "PL",
    "AU": "AU", "AUSTRALIA": "AU",
    "NZ": "NZ", "NEW ZEALAND": "NZ",
    "SG": "SG", "SINGAPORE": "SG",
    "JP": "JP", "JAPAN": "JP",
    "IN": "IN", "INDIA": "IN",
}  # fmt: skip

_GROUNDING_PERCENTILE = 0.01  # keep >= 99% of correct calib outputs passing
_GROUNDING_FLOOR = 80.0
_GROUNDING_CEIL = 100.0


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str | None = None


@dataclass(frozen=True)
class GateThresholds:
    """Every number a gate needs that was tuned on ``calib``, keyed by task name."""

    extraction_grounding_ratio: dict[str, float] = field(default_factory=dict)
    entity_matching_band: dict[str, tuple[float, float]] = field(default_factory=dict)
    entity_matching_threshold: dict[str, float] = field(default_factory=dict)
    classification_gate_enabled: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "extraction_grounding_ratio": dict(self.extraction_grounding_ratio),
            "entity_matching_band": {k: list(v) for k, v in self.entity_matching_band.items()},
            "entity_matching_threshold": dict(self.entity_matching_threshold),
            "classification_gate_enabled": dict(self.classification_gate_enabled),
        }


@dataclass
class GateContext:
    """Baselines and tuned thresholds a gate needs; built once from a run's ``calib`` data."""

    tfidf: dict[str, TfidfBaseline]
    fuzzy: dict[str, FuzzyMatchBaseline]
    regex_pii: RegexPiiBaseline
    thresholds: GateThresholds
    country_table: dict[str, str] = field(default_factory=lambda: dict(COUNTRY_TABLE))


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _partial_ratio(value: str, source: str) -> float:
    return float(fuzz.partial_ratio(_normalize_ws(value).casefold(), _normalize_ws(source).casefold()))


def _grounded(value: str, source: str, threshold: float) -> bool:
    norm_value = _normalize_ws(value).casefold()
    if not norm_value:
        return False
    norm_source = _normalize_ws(source).casefold()
    if norm_value in norm_source:
        return True
    return _partial_ratio(value, source) >= threshold


_PHONE_SUFFIX_LEN = 7  # enough digits that a coincidental match in unrelated text is implausible


def _digits_in_source(value: str, source: str) -> bool:
    """Whether ``value``'s digits appear in ``source``'s digit stream.

    A phone field is E.164 (with a country code) while the source text often writes the number in local
    format (no country code, or a different trunk prefix), so a full-string match would miss it; matching on
    the trailing subscriber digits handles both without requiring exact prefix agreement.
    """
    digits = re.sub(r"\D", "", value)
    if not digits:
        return False
    source_digits = re.sub(r"\D", "", source)
    if digits in source_digits:
        return True
    return len(digits) >= _PHONE_SUFFIX_LEN and digits[-_PHONE_SUFFIX_LEN:] in source_digits


def _country_in_source(code: str, source: str, table: dict[str, str]) -> bool:
    code_norm = code.strip().upper()
    if not code_norm:
        return False
    upper_source = source.upper()
    if re.search(rf"\b{re.escape(code_norm)}\b", upper_source):
        return True
    return any(mapped == code_norm and name in upper_source for name, mapped in table.items())


def _month_index(name: str) -> int | None:
    return _MONTHS_FULL.get(name.capitalize()) or _MONTHS_ABBR.get(name[:3].capitalize())


def resolve_next_weekday(ref: date, weekday_idx: int) -> date:
    days_ahead = (weekday_idx - ref.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return ref + timedelta(days=days_ahead)


def resolve_this_weekday(ref: date, weekday_idx: int) -> date:
    days_ahead = (weekday_idx - ref.weekday()) % 7
    return ref + timedelta(days=days_ahead)


def resolve_on_day_of_month(ref: date, day: int) -> date | None:
    year, month = ref.year, ref.month
    candidate: date | None
    try:
        candidate = date(year, month, day)
    except ValueError:
        candidate = None
    if candidate is None or candidate <= ref:
        month2, year2 = (month + 1, year) if month < 12 else (1, year + 1)
        try:
            candidate = date(year2, month2, day)
        except ValueError:
            return None
    return candidate


def resolve_date_expressions(source: str, reference_date: str) -> set[str]:
    """Every ISO date a human-readable expression in ``source`` can resolve to, given ``reference_date``.

    Covers ISO dates, ``DD/MM/YYYY``, ``14 October [2026]``, ``October 14[, 2026]``, weekday names with
    "next"/"this", "tomorrow", "in <n> days/weeks" (digits or small number words) and "on the 14th" -- the
    same resolution rules ``scripts/build_datasets.py`` used to derive extraction gold, so a correct answer on
    the bundled dataset always re-derives.
    """
    try:
        ref = date.fromisoformat(str(reference_date)[:10])
    except ValueError:
        return set()

    found: set[str] = set()
    found.update(m.group(0) for m in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", source))

    for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", source):
        day_num, month_num, year_num = (int(x) for x in m.groups())
        with contextlib.suppress(ValueError):
            found.add(date(year_num, month_num, day_num).isoformat())

    for m in re.finditer(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_NAME_RE})\.?,?\s*(\d{{4}})?\b", source, re.IGNORECASE
    ):
        day_s, mon, year_s = m.groups()
        month_idx = _month_index(mon)
        if month_idx:
            with contextlib.suppress(ValueError):
                found.add(date(int(year_s) if year_s else ref.year, month_idx, int(day_s)).isoformat())

    for m in re.finditer(
        rf"\b({_MONTH_NAME_RE})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s*(\d{{4}})?\b", source, re.IGNORECASE
    ):
        mon, day_s, year_s = m.groups()
        month_idx = _month_index(mon)
        if month_idx:
            with contextlib.suppress(ValueError):
                found.add(date(int(year_s) if year_s else ref.year, month_idx, int(day_s)).isoformat())

    for m in re.finditer(
        r"\b(next|this)\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", source, re.IGNORECASE
    ):
        which, weekday_name = m.groups()
        idx = _WEEKDAYS.index(weekday_name.capitalize())
        resolved = resolve_next_weekday(ref, idx) if which.lower() == "next" else resolve_this_weekday(ref, idx)
        found.add(resolved.isoformat())

    if re.search(r"\btomorrow\b", source, re.IGNORECASE):
        found.add((ref + timedelta(days=1)).isoformat())

    for m in re.finditer(r"\bin\s+([a-zA-Z]+|\d+)\s+(day|days|week|weeks)\b", source, re.IGNORECASE):
        num_text, unit = m.groups()
        n = int(num_text) if num_text.isdigit() else _WORD_NUMBERS.get(num_text.lower())
        if n:
            delta = timedelta(weeks=n) if unit.lower().startswith("week") else timedelta(days=n)
            found.add((ref + delta).isoformat())

    for m in re.finditer(r"\bon the (\d{1,2})(?:st|nd|rd|th)\b", source, re.IGNORECASE):
        resolved_day = resolve_on_day_of_month(ref, int(m.group(1)))
        if resolved_day is not None:
            found.add(resolved_day.isoformat())

    return found


def _date_rederivable(value: str, source: str, reference_date: str) -> bool:
    from local_enough.tasks.extraction import normalize_date

    target = normalize_date(value)
    return target in resolve_date_expressions(source, reference_date)


# -- threshold tuning (calib only) -------------------------------------------------------------------------


def _tune_extraction_grounding(run_dir: RunDir, spec: TaskSpec) -> float:
    fields = spec.fields or {}
    grounded_fields = [name for name, fs in fields.items() if fs.grounded]
    if not grounded_fields:
        return _GROUNDING_CEIL

    from local_enough.tasks import extraction

    items_by_id = {str(it["id"]): it for it in registry.load_items(spec, "calib")}
    ratios: list[float] = []
    for rec in run_dir.predictions("A"):
        if rec.get("task") != spec.name or rec.get("split") != "calib":
            continue
        item = items_by_id.get(str(rec.get("item_id")))
        if item is None:
            continue
        parsed = extraction.parse(spec, str(rec.get("raw") or ""))
        if not parsed.ok:
            continue
        gold = item.get("gold", {})
        source = str(item.get("text", ""))
        for name in grounded_fields:
            pred_raw = parsed.value.get(name)
            if pred_raw is None:
                continue
            fspec = fields[name]
            if normalize_field(fspec.type, gold.get(name), fspec) != normalize_field(fspec.type, pred_raw, fspec):
                continue
            ratios.append(_partial_ratio(str(pred_raw), source))

    if not ratios:
        return _GROUNDING_FLOOR
    ordered = sorted(ratios)
    n = len(ordered)
    keep_index = min(n - 1, math.floor(n * _GROUNDING_PERCENTILE))
    return min(_GROUNDING_CEIL, max(_GROUNDING_FLOOR, ordered[keep_index]))


def build_gate_context(run: RunDir | str, specs: dict[str, TaskSpec]) -> GateContext:
    """Rebuild the gate context (baselines + tuned thresholds) from a run's ``calib`` data.

    Deterministic given the same run and specs: ``route.planner`` calls this once to store the thresholds in
    the plan, and ``route.simulate``/``route.server`` call it again to get the same numbers back, so gate
    behaviour never depends on which of the two called it first.
    """
    run_dir = run if isinstance(run, RunDir) else RunDir(run)
    tfidf: dict[str, TfidfBaseline] = {}
    fuzzy: dict[str, FuzzyMatchBaseline] = {}
    grounding: dict[str, float] = {}
    band: dict[str, tuple[float, float]] = {}
    threshold: dict[str, float] = {}
    classification_enabled: dict[str, bool] = {}

    for name, spec in specs.items():
        if spec.kind == "classification":
            available = spec.train is not None and spec.split_path("train").exists()
            classification_enabled[name] = available
            if available:
                tfidf[name] = TfidfBaseline(registry.load_items(spec, "train"))
        elif spec.kind == "entity_matching":
            baseline = FuzzyMatchBaseline()
            baseline.tune(registry.load_items(spec, "calib"))
            fuzzy[name] = baseline
            if baseline.threshold is not None:
                threshold[name] = baseline.threshold
            if baseline.band is not None:
                band[name] = baseline.band
        elif spec.kind == "extraction":
            grounding[name] = _tune_extraction_grounding(run_dir, spec)

    thresholds = GateThresholds(
        extraction_grounding_ratio=grounding,
        entity_matching_band=band,
        entity_matching_threshold=threshold,
        classification_gate_enabled=classification_enabled,
    )
    return GateContext(tfidf=tfidf, fuzzy=fuzzy, regex_pii=RegexPiiBaseline(), thresholds=thresholds)


# -- the gate itself ----------------------------------------------------------------------------------------


def gate(spec: TaskSpec, item: Item, parsed: Parsed, ctx: GateContext, *, format_only: bool = False) -> GateResult:
    """Spec item 16's deterministic evidence check for one predicted answer.

    ``item`` is the router's own view of the request (no gold: what ``item_from_input`` builds, or a dataset
    row when replaying). ``format_only`` is set when the candidate being gated is itself the baseline a gate
    would otherwise compare against (the plan marks this on its primary); the comparison step is then skipped
    since it would trivially agree with itself.
    """
    if spec.kind == "classification":
        return _gate_classification(spec, item, parsed, ctx, format_only)
    if spec.kind == "extraction":
        return _gate_extraction(spec, item, parsed, ctx)
    if spec.kind == "pii_redaction":
        return _gate_pii(item, parsed, ctx, format_only)
    if spec.kind == "summarisation":
        return _gate_summarisation(spec, item, parsed)
    if spec.kind == "entity_matching":
        return _gate_entity_matching(spec, item, parsed, ctx, format_only)
    raise ValueError(f"unknown task kind {spec.kind!r}")


def _gate_classification(spec: TaskSpec, item: Item, parsed: Parsed, ctx: GateContext, format_only: bool) -> GateResult:
    if not parsed.ok:
        return GateResult(False, "invalid_label")
    label = str(parsed.value)
    if spec.labels and label not in spec.labels:
        return GateResult(False, "label_not_in_set")
    if format_only:
        return GateResult(True)
    baseline = ctx.tfidf.get(spec.name)
    if baseline is None:
        return GateResult(True, "baseline_unavailable")
    if baseline.predict(str(item.get("text", ""))) != label:
        return GateResult(False, "disagrees_with_baseline")
    return GateResult(True)


def _gate_extraction(spec: TaskSpec, item: Item, parsed: Parsed, ctx: GateContext) -> GateResult:
    if not parsed.ok:
        return GateResult(False, "invalid_json")
    fields = spec.fields or {}
    pred = parsed.value or {}
    source = str(item.get("text", ""))
    reference_date = item.get("reference_date")
    threshold = ctx.thresholds.extraction_grounding_ratio.get(spec.name, _GROUNDING_FLOOR)

    for name, fspec in fields.items():
        value = pred.get(name)
        if fspec.required and value is None:
            return GateResult(False, f"missing_required:{name}")
        if value is None:
            continue
        failure = _extraction_field_failure(fspec, name, str(value), source, reference_date, threshold, ctx)
        if failure is not None:
            return GateResult(False, failure)
    return GateResult(True)


def _extraction_field_failure(
    fspec: FieldSpec, name: str, value: str, source: str, reference_date: Any, threshold: float, ctx: GateContext
) -> str | None:
    """The gate-failure reason for one normalised, non-null extraction field, or ``None`` when it passes."""
    if fspec.grounded:
        return None if _grounded(value, source, threshold) else f"ungrounded:{name}"
    if fspec.type == "phone":
        return None if _digits_in_source(value, source) else f"phone_not_in_source:{name}"
    if fspec.type == "country":
        return None if _country_in_source(value, source, ctx.country_table) else f"country_not_in_source:{name}"
    if fspec.type == "date":
        if reference_date and not _date_rederivable(value, source, str(reference_date)):
            return f"date_not_rederivable:{name}"
        return None
    if fspec.type == "enum" and fspec.values:
        allowed = {v.casefold() for v in fspec.values}
        return None if value.strip().casefold() in allowed else f"not_in_enum:{name}"
    if fspec.type == "amount":
        return None if _digits_in_source(value, source) else f"amount_not_in_source:{name}"
    return None


def _gate_pii(item: Item, parsed: Parsed, ctx: GateContext, format_only: bool) -> GateResult:
    if not parsed.ok:
        return GateResult(False, "invalid_json")
    if format_only:
        return GateResult(True)
    text = str(item.get("text", ""))
    predicted = parsed.value or []
    norm_text = _normalize_ws(text).casefold()
    for entry in predicted:
        span_text = _normalize_ws(str(entry.get("text", ""))).casefold()
        if not span_text or span_text not in norm_text:
            return GateResult(False, "predicted_span_not_in_text")

    predicted_norm = [_normalize_ws(str(e.get("text", ""))).casefold() for e in predicted]
    for baseline_span in ctx.regex_pii.find(text):
        baseline_norm = _normalize_ws(baseline_span["text"]).casefold()
        covered = any(baseline_norm in p or p in baseline_norm for p in predicted_norm if p)
        if not covered:
            return GateResult(False, "regex_baseline_span_missed")
    return GateResult(True)


_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SUMMARY_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{4}\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTH_NAME_RE})\.?,?\s*\d{{0,4}}\b"
    rf"|\b(?:{_MONTH_NAME_RE})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s*\d{{0,4}}\b",
    re.IGNORECASE,
)


def _gate_summarisation(spec: TaskSpec, item: Item, parsed: Parsed) -> GateResult:
    if not parsed.ok:
        return GateResult(False, "empty_summary")
    summary = str(parsed.value)
    max_words = item.get("max_words") or spec.max_words or 120
    if len(summary.split()) > max_words:
        return GateResult(False, "over_word_limit")

    source = str(item.get("text", ""))
    source_digits = re.sub(r"\D", "", source)
    for match in _NUMBER_RE.finditer(summary):
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) >= 2 and digits not in source_digits:
            return GateResult(False, "number_not_in_source")

    norm_source = _normalize_ws(source).casefold()
    for match in _SUMMARY_DATE_RE.finditer(summary):
        if _normalize_ws(match.group(0)).casefold() not in norm_source:
            return GateResult(False, "date_not_in_source")
    return GateResult(True)


def _gate_entity_matching(
    spec: TaskSpec, item: Item, parsed: Parsed, ctx: GateContext, format_only: bool
) -> GateResult:
    if not parsed.ok or not isinstance(parsed.value, dict):
        return GateResult(False, "invalid_match_json")
    if format_only:
        return GateResult(True)
    baseline = ctx.fuzzy.get(spec.name)
    if baseline is None or baseline.threshold is None:
        return GateResult(True, "baseline_unavailable")
    score = baseline.similarity(item.get("left", {}), item.get("right", {}))
    lo, hi = baseline.band or (baseline.threshold, baseline.threshold)
    if lo <= score <= hi:
        return GateResult(True, "inside_uncertainty_band")
    predicted_match = bool(parsed.value.get("match"))
    baseline_match = score >= baseline.threshold
    if predicted_match != baseline_match:
        return GateResult(False, "disagrees_with_baseline")
    return GateResult(True)
