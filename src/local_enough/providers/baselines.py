"""Three non-LLM baselines: TF-IDF classification, regex PII, and fuzzy entity matching.

Each baseline takes plain dicts and lists -- never ``local_enough.tasks.registry`` -- and exposes
``respond(item_input) -> str`` in the same output format an LLM must produce, so the task module's
own parser and scorer apply unchanged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from local_enough.config import BaselineModel
from local_enough.tasks.base import TaskSpec

_RANDOM_STATE = 7


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                items.append(json.loads(stripped))
    return items


class TfidfBaseline:
    """Word 1-2 gram TF-IDF plus logistic regression, trained on a task's ``train`` split."""

    def __init__(self, train: list[dict[str, Any]]) -> None:
        if not train:
            raise ValueError("TfidfBaseline requires a non-empty train split")
        texts = [str(item["text"]) for item in train]
        labels = [str(item["label"]) for item in train]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer="word")
        matrix = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(random_state=_RANDOM_STATE, max_iter=1000)
        self.model.fit(matrix, labels)

    def predict(self, text: str) -> str:
        matrix = self.vectorizer.transform([text])
        return str(self.model.predict(matrix)[0])

    def respond(self, item_input: str) -> str:
        return self.predict(item_input)


_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

_PHONE_RE = re.compile(
    r"\+44[\s-]?7700[\s-]?9\d{2}[\s-]?\d{3}\b"  # UK intl fictional mobile
    r"|\b07700[\s-]?9\d{2}[\s-]?\d{3}\b"  # UK fictional mobile
    r"|\(?\d{3}\)?[\s.-]?555[\s.-]?01\d{2}\b"  # US/CA fictional
    r"|\+\d{1,3}[\s-]?\(?\d{2,4}\)?(?:[\s-]?\d{2,4}){2,4}\b"  # generic international
)

_IBAN_CANDIDATE_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")

_DOB_CUE_RE = re.compile(r"born|dob|date of birth", re.IGNORECASE)
_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{4}\b"
    r"|\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{4}\b",
    re.IGNORECASE,
)


def _iban_valid(candidate: str) -> bool:
    compact = candidate.replace(" ", "").upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    try:
        digits = "".join(str(int(ch, 36)) for ch in rearranged)
    except ValueError:
        return False
    return int(digits) % 97 == 1


def _find_dates_of_birth(text: str, window: int = 40) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for match in _DATE_RE.finditer(text):
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        if _DOB_CUE_RE.search(text[start:end]):
            hits.append({"type": "DATE_OF_BIRTH", "text": match.group(0)})
    return hits


class RegexPiiBaseline:
    """Regex spotting for EMAIL, PHONE, IBAN (mod-97 checked) and DATE_OF_BIRTH."""

    def find(self, text: str) -> list[dict[str, str]]:
        found: list[dict[str, str]] = []
        found.extend({"type": "EMAIL", "text": m.group(0)} for m in _EMAIL_RE.finditer(text))
        found.extend({"type": "PHONE", "text": m.group(0)} for m in _PHONE_RE.finditer(text))
        found.extend(
            {"type": "IBAN", "text": m.group(0)} for m in _IBAN_CANDIDATE_RE.finditer(text) if _iban_valid(m.group(0))
        )
        found.extend(_find_dates_of_birth(text))
        return found

    def respond(self, item_input: str) -> str:
        return json.dumps(self.find(item_input))


_FIELD_WEIGHTS: dict[str, float] = {
    "name": 0.30,
    "street": 0.15,
    "postcode": 0.15,
    "city": 0.10,
    "country": 0.10,
    "domain": 0.10,
    "vat_id": 0.05,
    "phone": 0.05,
}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _digits(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))


def _field_similarity(field_name: str, left_value: Any, right_value: Any) -> float:
    if field_name == "vat_id":
        left_norm, right_norm = _norm(left_value), _norm(right_value)
        return 1.0 if left_norm and left_norm == right_norm else 0.0
    if field_name == "phone":
        left_digits, right_digits = _digits(left_value), _digits(right_value)
        if not left_digits and not right_digits:
            return 0.0
        return fuzz.ratio(left_digits, right_digits) / 100.0
    if field_name == "name":
        return fuzz.token_set_ratio(_norm(left_value), _norm(right_value)) / 100.0
    return fuzz.ratio(_norm(left_value), _norm(right_value)) / 100.0


def _f1(scored: list[tuple[float, bool]], threshold: float) -> float:
    tp = sum(1 for score, match in scored if score >= threshold and match)
    fp = sum(1 for score, match in scored if score >= threshold and not match)
    fn = sum(1 for score, match in scored if score < threshold and match)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _accuracy(scored: list[tuple[float, bool]], threshold: float) -> float:
    if not scored:
        return 0.0
    correct = sum(1 for score, match in scored if (score >= threshold) == match)
    return correct / len(scored)


class FuzzyMatchBaseline:
    """Rapidfuzz field-similarity matcher with a threshold and uncertainty band tuned on calib."""

    def __init__(self) -> None:
        self.threshold: float | None = None
        self.band: tuple[float, float] | None = None

    def similarity(self, left: dict[str, Any], right: dict[str, Any]) -> float:
        return sum(
            weight * _field_similarity(field, left.get(field), right.get(field))
            for field, weight in _FIELD_WEIGHTS.items()
        )

    def score(self, item: dict[str, Any]) -> float:
        """Convenience wrapper over an item shaped ``{"left": {...}, "right": {...}}``."""
        return self.similarity(item["left"], item["right"])

    def tune(self, calib_pairs: list[dict[str, Any]]) -> None:
        """Pick the F1-maximising threshold; the uncertainty band spans thresholds whose own
        split accuracy on ``calib_pairs`` is below 0.9."""
        scored = [(self.similarity(pair["left"], pair["right"]), bool(pair["match"])) for pair in calib_pairs]
        candidates = sorted({round(score, 6) for score, _ in scored} | {0.0, 1.0})
        best_threshold = max(candidates, key=lambda t: (_f1(scored, t), -t)) if candidates else 0.5
        self.threshold = best_threshold
        uncertain = [t for t in candidates if _accuracy(scored, t) < 0.9]
        self.band = (min(uncertain), max(uncertain)) if uncertain else (best_threshold, best_threshold)

    def predict(self, left: dict[str, Any], right: dict[str, Any]) -> bool:
        if self.threshold is None:
            raise RuntimeError("FuzzyMatchBaseline.tune() must be called before predict()")
        return self.similarity(left, right) >= self.threshold

    def respond(self, item_input: str) -> str:
        payload = json.loads(item_input)
        return json.dumps({"match": self.predict(payload["left"], payload["right"])})


Baseline = TfidfBaseline | RegexPiiBaseline | FuzzyMatchBaseline


def baseline_for(model_cfg: BaselineModel, spec: TaskSpec) -> Baseline | None:
    """Build the baseline for a ``kind: baseline`` config entry from the task's own dataset files.

    Returns ``None`` when the baseline is not available for this task (for example a classification
    task with no ``train`` split).
    """
    if model_cfg.task == "classification":
        if spec.train is None:
            return None
        train_path = spec.split_path("train")
        if not train_path.exists():
            return None
        return TfidfBaseline(_read_jsonl(train_path))
    if model_cfg.task == "pii_redaction":
        return RegexPiiBaseline()
    if model_cfg.task == "entity_matching":
        baseline = FuzzyMatchBaseline()
        baseline.tune(_read_jsonl(spec.split_path("calib")))
        return baseline
    return None
