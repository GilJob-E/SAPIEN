"""Unit tests for _smoke_helpers — pure functions only, no I/O."""
from __future__ import annotations

import pytest

from scripts._smoke_helpers import (
    Rating,
    aggregate_ratings,
    compute_percentiles,
    format_pass_fail,
    word_error_rate,
)


class TestAggregateRatings:
    def test_single_rating_mean_is_rating(self):
        ratings = [Rating(fluency=4, relevance=5, register=3)]
        result = aggregate_ratings(ratings)
        assert result["mean"] == pytest.approx((4 + 5 + 3) / 3)

    def test_mean_across_multiple_ratings(self):
        ratings = [
            Rating(fluency=4, relevance=4, register=4),
            Rating(fluency=2, relevance=2, register=2),
        ]
        result = aggregate_ratings(ratings)
        assert result["mean"] == pytest.approx(3.0)

    def test_min_reports_lowest_individual_axis(self):
        ratings = [
            Rating(fluency=5, relevance=5, register=1),  # register drags min
            Rating(fluency=4, relevance=4, register=4),
        ]
        result = aggregate_ratings(ratings)
        assert result["min"] == 1

    def test_empty_ratings_returns_zero(self):
        result = aggregate_ratings([])
        assert result == {"mean": 0.0, "min": 0, "count": 0}


class TestComputePercentiles:
    def test_single_value_all_percentiles_equal(self):
        result = compute_percentiles([1.5])
        assert result["p50"] == pytest.approx(1.5)
        assert result["p95"] == pytest.approx(1.5)
        assert result["p99"] == pytest.approx(1.5)

    def test_sorted_values_p50_is_median(self):
        # 5 values — p50 is middle
        result = compute_percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["p50"] == pytest.approx(3.0)

    def test_p95_captures_upper_tail(self):
        values = list(range(1, 101))  # 1..100
        result = compute_percentiles(values)
        assert result["p95"] == pytest.approx(95.05, abs=0.5)

    def test_empty_values_returns_zero(self):
        result = compute_percentiles([])
        assert result == {"p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0}


class TestWordErrorRate:
    def test_identical_transcripts_zero_wer(self):
        assert word_error_rate("안녕하세요 반갑습니다", "안녕하세요 반갑습니다") == 0.0

    def test_one_word_wrong_nonzero_wer(self):
        # 2 words total, 1 substitution → 0.5
        wer = word_error_rate("안녕하세요 반갑습니다", "안녕하세요 만났어요")
        assert wer == pytest.approx(0.5)

    def test_empty_reference_returns_one(self):
        # fully wrong, no reference → WER 1.0 (fallback)
        assert word_error_rate("", "뭐든지") == 1.0


class TestFormatPassFail:
    def test_pass_format(self):
        assert format_pass_fail(True) == "✅ PASS"

    def test_fail_format(self):
        assert format_pass_fail(False) == "❌ FAIL"
