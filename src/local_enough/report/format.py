"""Fixed, deterministic number formatting shared by every report table and the headline.

Every function returns a plain string, never locale-dependent, so the same run directory renders
byte-identical output on any machine. ``NaN``/``inf``/``None`` all render as a short, explicit label
instead of Python's own spelling, so a reader never has to guess what "nan" means in a table cell.
"""

from __future__ import annotations

import math

NOT_MEASURED = "not measured"
NOT_JUDGED = "not judged"
NA = "n/a"


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def fmt_pct(value: float | None, *, decimals: int = 1, missing: str = NA) -> str:
    """A fraction in ``[0, 1]`` (or outside it) as a percentage, e.g. ``0.953 -> "95.3%"``."""
    v = _finite(value)
    return missing if v is None else f"{v * 100:.{decimals}f}%"


def fmt_signed_pct(value: float | None, *, decimals: int = 1, missing: str = NA) -> str:
    """Like :func:`fmt_pct` but always shows a sign, for savings/deltas."""
    v = _finite(value)
    return missing if v is None else f"{v * 100:+.{decimals}f}%"


def fmt_ci_pct(lo: float | None, hi: float | None, *, decimals: int = 1) -> str:
    lo_s, hi_s = fmt_pct(lo, decimals=decimals), fmt_pct(hi, decimals=decimals)
    if lo_s == NA or hi_s == NA:
        return NA
    return f"[{lo_s}, {hi_s}]"


def fmt_usd(value: float | None, *, decimals: int = 2, missing: str = NA) -> str:
    v = _finite(value)
    return missing if v is None else f"${v:,.{decimals}f}"


def fmt_usd_per_1k(usd_per_task: float | None, *, missing: str = NA) -> str:
    """USD per task rendered as USD per 1,000 tasks, the unit every report table uses."""
    v = _finite(usd_per_task)
    if v is None:
        return missing
    per_1k = v * 1000.0
    decimals = 2 if per_1k >= 0.01 else 4
    return f"${per_1k:,.{decimals}f}"


def fmt_number(value: float | None, *, decimals: int = 0, missing: str = NA) -> str:
    v = _finite(value)
    return missing if v is None else f"{v:,.{decimals}f}"


def fmt_int(value: int | float | None, *, missing: str = NA) -> str:
    v = _finite(float(value)) if value is not None else None
    return missing if v is None else f"{round(v):,}"


def fmt_seconds(value: float | None, *, decimals: int = 2, missing: str = NOT_MEASURED) -> str:
    v = _finite(value)
    return missing if v is None else f"{v:.{decimals}f}s"


def fmt_gb(bytes_value: float | None, *, decimals: int = 2, missing: str = NA) -> str:
    v = _finite(bytes_value)
    return missing if v is None else f"{v / 1_000_000_000:.{decimals}f} GB"


def fmt_gb_whole(gb: float | None, *, missing: str = NA) -> str:
    """A memory size already in GB, without a trailing ``.00``: ``128.0 -> "128 GB"``, ``24.5 -> "24.5 GB"``."""
    v = _finite(gb)
    if v is None:
        return missing
    return f"{v:,.0f} GB" if float(v).is_integer() else f"{v:,.2f} GB"


def fmt_gib(bytes_value: float | None, *, decimals: int = 2, missing: str = NA) -> str:
    """Bytes as GiB (2**30), the unit macOS reports memory in."""
    v = _finite(bytes_value)
    return missing if v is None else f"{v / 2**30:.{decimals}f} GiB"


def fmt_ms(value: float | None, *, decimals: int = 0, missing: str = NA) -> str:
    v = _finite(value)
    return missing if v is None else f"{v:.{decimals}f} ms"


def fmt_watts(value: float | None, *, decimals: int = 1, missing: str = NOT_MEASURED) -> str:
    v = _finite(value)
    return missing if v is None else f"{v:.{decimals}f} W"


def fmt_bool(value: bool | None, *, missing: str = NA) -> str:
    if value is None:
        return missing
    return "yes" if value else "no"
