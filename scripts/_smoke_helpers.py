"""
_smoke_helpers.py — Pure functions shared by the three smoke-gate scripts.

NO network calls, NO model loads, NO file I/O from these functions.
All I/O happens in the gate scripts themselves so unit tests run fast and offline.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence

import jiwer


@dataclass(frozen=True)
class Rating:
    """Single rater's score on one prompt (1-5 on each axis)."""
    fluency: int
    relevance: int
    register: int

    def __post_init__(self) -> None:
        for axis, value in [
            ("fluency", self.fluency),
            ("relevance", self.relevance),
            ("register", self.register),
        ]:
            if not 1 <= value <= 5:
                raise ValueError(f"{axis}={value} outside 1..5")


def aggregate_ratings(ratings: Sequence[Rating]) -> dict:
    """Return mean across all axes + minimum individual axis score + count.

    Empty input returns {"mean": 0.0, "min": 0, "count": 0} to avoid raising
    in the CLI happy path (the gate script logs 0-count separately).
    """
    if not ratings:
        return {"mean": 0.0, "min": 0, "count": 0}

    all_values: list[int] = []
    for r in ratings:
        all_values.extend([r.fluency, r.relevance, r.register])

    return {
        "mean": statistics.fmean(all_values),
        "min": min(all_values),
        "count": len(ratings),
    }


def compute_percentiles(values: Iterable[float]) -> dict:
    """p50/p95/p99 over a sequence of numeric values. Empty → zeros."""
    data = sorted(values)
    if not data:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0}

    def pct(p: float) -> float:
        # Linear interpolation between closest ranks.
        idx = (len(data) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(data) - 1)
        frac = idx - lo
        return data[lo] + frac * (data[hi] - data[lo])

    return {
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "count": len(data),
    }


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER for Korean transcription comparison. Empty reference → 1.0."""
    if not reference.strip():
        return 1.0
    return float(jiwer.wer(reference, hypothesis))


def format_pass_fail(passed: bool) -> str:
    """Canonical emoji-prefixed pass/fail label for Markdown output."""
    return "✅ PASS" if passed else "❌ FAIL"
