# M0 — Pre-migration Smoke Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate three GO/NO-GO gates before committing to the full Parlor-style hybrid migration: (1) Gemma 4 E2B Korean dialogue quality, (2) Gemma 4 E2B multimodal (audio+vision) capability, (3) M2 MacBook end-to-end latency. If any gate fails, the migration is halted with a documented decision.

**Architecture:** Pure validation scripts under `scripts/` — zero changes to production (`start_app/`). Three gate scripts plus shared helpers, a small fixtures directory of Korean test data, and a consolidated results document. No FastAPI, no `start_app_ws/`, no browser changes — this milestone produces only validation artifacts.

**Tech Stack:** Python 3.12, LiteRT-LM (pip package from Google AI Edge), Gemma 4 E2B model from HuggingFace `litert-community/gemma-4-E2B-it-litert-lm`, existing Gemini 2.0 Flash (cloud) as baseline via the project's existing `start_app/dialogue_manager/llm.py`, `pytest` for helper unit tests.

**Design spec:** `docs/superpowers/specs/2026-04-17-parlor-hybrid-migration-design.md` (especially §8.1).

---

## File Structure

### Created by this plan

```
scripts/
├── __init__.py                               # package marker
├── _smoke_helpers.py                         # shared utilities (rating aggregation, WER, percentiles)
├── smoke_gemma_korean.py                     # Gate #1
├── smoke_gemma_multimodal.py                 # Gate #2
├── bench_latency.py                          # Gate #3
├── smoke_results.md                          # consolidated decision document (final artifact)
├── fixtures/
│   ├── korean_prompts.json                   # 10 Korean interview prompts
│   ├── expected_emotions.json                # ground-truth labels for Gate #2
│   ├── audio_samples/                        # 5 recorded WAV files (manual record)
│   │   ├── self_intro.wav
│   │   ├── motivation.wav
│   │   ├── experience.wav
│   │   ├── weakness.wav
│   │   └── question.wav
│   └── test_photo.jpg                        # single neutral photo for Gate #2
└── tests/
    ├── __init__.py
    ├── test_smoke_helpers.py                 # unit tests for _smoke_helpers.py
    ├── test_korean_prompts_fixture.py        # structural test of fixture JSON
    └── fixtures/
        └── mock_samples/                     # tiny synthetic fixtures for unit tests

requirements-smoke.txt                        # pip deps for smoke tests only (kept isolated)
```

### Modified by this plan

```
.gitignore                                    # ignore fixtures/audio_samples/*.wav if large, and models/
README.md                                     # add "Smoke Tests (Validation)" section
```

### File responsibilities

- `_smoke_helpers.py` — pure functions, no I/O to external services. Rating aggregation, WER calculation, percentile stats, JSON/Markdown output formatting. **All unit-testable.**
- `smoke_gemma_korean.py` — CLI script: loads Gemma, loads Gemini baseline, iterates prompts, captures ratings interactively, writes JSON results.
- `smoke_gemma_multimodal.py` — CLI script: loads Gemma with audio+vision, iterates audio+photo pairs, computes WER and emotion match, writes JSON results.
- `bench_latency.py` — CLI script: timed loop of Gemma inference + simulated ElevenLabs first-byte, computes percentiles, writes JSON results.
- `smoke_results.md` — human-written summary after running all three gates, with GO/NO-GO decision.
- `korean_prompts.json` — structured data, checked into git for reproducibility.

---

## Prerequisites

### P0: Step 5-1 branch landed

The current branch `step5-1/latency-optimization` has uncommitted changes to `start_app/dialogue_manager/meeting.py` that must be finalized before M0 starts. Per spec §9.1, the Legacy path must be stabilized first.

- [ ] **Step 1: Inspect uncommitted diff on step5-1/latency-optimization**

Run: `cd /Users/hoddukzoa/Desktop/학교/giljob-e/SAPIEN && git status && git diff start_app/dialogue_manager/meeting.py`

Expected: you see the `finish_reason` debug print, concise-response system message, `max_tokens=500`, and Korean punctuation guard.

- [ ] **Step 2: Remove the debug `print` from `gemini_chat_call`**

Modify `start_app/dialogue_manager/meeting.py` lines near the `gemini_chat_call` function. Remove exactly these 3 added lines (they were added for debugging, not for production):

```python
    if response.candidates:
        fr = response.candidates[0].finish_reason
        print(f"[LLM] finish_reason={fr}, max_tokens={max_tokens}, len={len(response.text)}")
```

Keep the other changes (concise response system message, max_tokens=500, language guard).

- [ ] **Step 3: Stage and commit the cleaned Step 5-1 changes**

```bash
git add start_app/dialogue_manager/meeting.py
# Also remove the stale vim undo files that were deleted locally
git add -u start_app/static/js/
git commit -m "chore(step5-1): remove debug print, finalize latency tuning

- finish_reason debug print 제거 (개발 중 임시 삽입)
- 기존 변경사항 유지: concise system message, max_tokens=500, 언어별 문장 가드
- 레거시 경로 안정화 완료 (Parlor 마이그레이션의 폴백 경로)"
```

- [ ] **Step 4: Push step5-1 branch and open PR (manual via gh CLI)**

```bash
git push -u origin step5-1/latency-optimization
~/bin/gh pr create --fill --base main --head step5-1/latency-optimization
```

Then merge the PR via GitHub (manual user action). Close issue #15 with a "completed" comment.

- [ ] **Step 5: Verify main is updated and create new feature branch**

```bash
git checkout main
git pull origin main
git checkout -b parlor-hybrid/m0-smoke-gates
git push -u origin parlor-hybrid/m0-smoke-gates
```

Expected: `git log -1 --oneline` on `main` shows the merged Step 5-1 commit.

---

## Task 1: LiteRT-LM Environment Validation

**Files:**
- Create: `requirements-smoke.txt`
- Create: `scripts/__init__.py`
- Create: `scripts/_env_check.py` (temporary — will be deleted at end of Task 1)

This task validates that LiteRT-LM + Gemma 4 E2B actually runs on M2 base hardware. If this fails, the migration is blocked pending LiteRT-LM fix or Ollama alternative. This is the riskiest single step in M0 — done first to fail fast.

- [ ] **Step 1: Create `requirements-smoke.txt` with smoke-test-only dependencies**

Create file at repo root:

```
# Smoke test dependencies (M0 gate scripts only)
# DO NOT install these into production environment — they are isolated per design spec §4.2
litert-lm>=0.4.0
numpy>=1.26
Pillow>=10.0
pytest>=8.0
pytest-timeout>=2.3
jiwer>=3.0          # for WER calculation
google-generativeai>=0.8  # existing Gemini baseline (already in requirements.txt)
rich>=13.0          # nice CLI formatting
```

- [ ] **Step 2: Install smoke-test dependencies in an isolated venv**

```bash
cd /Users/hoddukzoa/Desktop/학교/giljob-e/SAPIEN
python3.12 -m venv .venv-smoke
source .venv-smoke/bin/activate
pip install --upgrade pip
pip install -r requirements-smoke.txt
```

Expected: no errors. If `litert-lm` fails to install on your Python 3.12, check https://ai.google.dev/edge/litert-lm/overview for current wheels. If wheels are unavailable for Apple Silicon Python 3.12, consult the spec §10 open questions — this is a migration blocker.

- [ ] **Step 3: Download the Gemma 4 E2B LiteRT-LM model**

```bash
mkdir -p models/gemma-4-e2b
cd models/gemma-4-e2b
# Model is 2.58 GB — download takes a few minutes
curl -L -o gemma-4-E2B-it.litertlm \
  "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm"
cd ../..
```

Expected: file `models/gemma-4-e2b/gemma-4-E2B-it.litertlm` is ~2.58 GB. If the HuggingFace URL path differs, go to https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm and copy the current download link.

- [ ] **Step 4: Add `models/` to .gitignore**

```bash
echo "" >> .gitignore
echo "# Local ML models (Parlor-hybrid migration)" >> .gitignore
echo "models/" >> .gitignore
echo ".venv-smoke/" >> .gitignore
```

- [ ] **Step 5: Create `scripts/__init__.py`**

```python
# scripts/__init__.py
"""M0 smoke test scripts — not production code."""
```

- [ ] **Step 6: Write a minimal hello-world LiteRT-LM script**

Create `scripts/_env_check.py`:

```python
"""
_env_check.py — Validate LiteRT-LM + Gemma 4 E2B on this machine.

Exit code 0 means the environment is ready for smoke tests.
This file is deleted at the end of Task 1.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "gemma-4-e2b" / "gemma-4-E2B-it.litertlm"


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"[FAIL] Model not found at {MODEL_PATH}")
        print("       Re-run Task 1 Step 3.")
        return 1

    try:
        from litert_lm import LlmInference  # type: ignore
    except ImportError as exc:
        print(f"[FAIL] litert-lm not importable: {exc}")
        print("       Re-run Task 1 Step 2.")
        return 2

    print("[1/3] Loading model (3GB, ~20s)...")
    t0 = time.perf_counter()
    engine = LlmInference.create_from_file(str(MODEL_PATH))
    load_s = time.perf_counter() - t0
    print(f"      loaded in {load_s:.1f}s")

    print("[2/3] Running warmup inference...")
    t0 = time.perf_counter()
    out = engine.generate_response("안녕하세요. 자기소개 부탁드립니다.")
    warm_s = time.perf_counter() - t0
    print(f"      generated in {warm_s:.1f}s")
    print(f"      response preview: {out[:120]!r}")

    print("[3/3] Verifying Korean output is coherent...")
    if len(out) < 20:
        print(f"[FAIL] response too short ({len(out)} chars); model may be misconfigured")
        return 3
    if "안녕" not in out and "감사" not in out and "저는" not in out:
        print("[WARN] response does not contain common Korean greeting tokens;")
        print("       manual inspection required — proceed with caution.")

    print("[OK] environment is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

> **Note:** The LiteRT-LM Python API name (`LlmInference.create_from_file`, `generate_response`) is based on Google AI Edge's documented pattern (https://ai.google.dev/edge/litert-lm/overview). If the installed `litert-lm` version exposes a different name (e.g. `Engine.load`, `engine.generate`), update the two `import`/method calls accordingly. The structure of the check (load → warmup → verify) is stable regardless of exact naming.

- [ ] **Step 7: Run the environment check**

```bash
source .venv-smoke/bin/activate
python scripts/_env_check.py
```

Expected output ends with `[OK] environment is ready.` If any step fails, **halt the plan** and consult the linked LiteRT-LM docs. Do not proceed to Task 2 until this passes.

- [ ] **Step 8: Delete the env check file and commit progress**

```bash
rm scripts/_env_check.py
git add requirements-smoke.txt scripts/__init__.py .gitignore
git commit -m "chore(m0): add smoke-test isolated env, gitignore models/

- requirements-smoke.txt: LiteRT-LM + deps (isolated from production)
- .venv-smoke/ gitignored
- models/ gitignored (Gemma 4 E2B .litertlm is 2.58 GB)
- scripts/ package scaffold

Environment validated manually — Gemma 4 E2B loads and generates Korean
responses on M2 MacBook base."
```

---

## Task 2: Shared Helpers with Unit Tests

**Files:**
- Create: `scripts/_smoke_helpers.py`
- Create: `scripts/tests/__init__.py`
- Create: `scripts/tests/test_smoke_helpers.py`

Pure functions used by all three gate scripts. Keeps Gemma/Gemini I/O out of the test path so unit tests run in CI without any model/API dependency.

- [ ] **Step 1: Write the failing test file**

Create `scripts/tests/__init__.py` (empty file).

Create `scripts/tests/test_smoke_helpers.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail (helpers not implemented yet)**

```bash
source .venv-smoke/bin/activate
python -m pytest scripts/tests/test_smoke_helpers.py -v
```

Expected: all tests fail with `ModuleNotFoundError: No module named 'scripts._smoke_helpers'` or `ImportError`.

- [ ] **Step 3: Implement `_smoke_helpers.py` to make tests pass**

Create `scripts/_smoke_helpers.py`:

```python
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
```

- [ ] **Step 4: Run tests again and verify all pass**

```bash
python -m pytest scripts/tests/test_smoke_helpers.py -v
```

Expected: 12 tests pass in < 1 second.

- [ ] **Step 5: Commit helpers + tests**

```bash
git add scripts/_smoke_helpers.py scripts/tests/__init__.py scripts/tests/test_smoke_helpers.py
git commit -m "test(m0): shared smoke-test helpers with unit tests

- Rating dataclass (1-5 on fluency/relevance/register axes)
- aggregate_ratings: mean + min across prompts
- compute_percentiles: p50/p95/p99 latency stats
- word_error_rate: jiwer-backed Korean transcription diff
- format_pass_fail: canonical markdown labels

12 unit tests, all pure functions, < 1s CI runtime."
```

---

## Task 3: Korean Prompts Fixture + Validation Test

**Files:**
- Create: `scripts/fixtures/korean_prompts.json`
- Create: `scripts/tests/test_korean_prompts_fixture.py`

The 10 prompts used by Gate #1. Stored as JSON for reproducibility. Test validates structure so a future accidental edit doesn't break the script silently.

- [ ] **Step 1: Write the failing fixture-structure test**

Create `scripts/tests/test_korean_prompts_fixture.py`:

```python
"""Structural validation of the Korean prompts fixture used by Gate #1."""
from __future__ import annotations

import json
from pathlib import Path

FIXTURE = Path(__file__).parent.parent / "fixtures" / "korean_prompts.json"


def test_fixture_exists():
    assert FIXTURE.exists(), f"Missing fixture: {FIXTURE}"


def test_fixture_is_valid_json():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


def test_fixture_has_exactly_ten_prompts():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert "prompts" in data
    assert len(data["prompts"]) == 10


def test_every_prompt_has_required_fields():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for i, p in enumerate(data["prompts"]):
        assert "id" in p, f"prompt[{i}] missing 'id'"
        assert "category" in p, f"prompt[{i}] missing 'category'"
        assert "text" in p, f"prompt[{i}] missing 'text'"
        assert len(p["text"]) >= 10, f"prompt[{i}] too short"


def test_prompt_ids_are_unique():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    ids = [p["id"] for p in data["prompts"]]
    assert len(ids) == len(set(ids)), "duplicate prompt ids"


def test_categories_cover_interview_breadth():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    categories = {p["category"] for p in data["prompts"]}
    # At minimum, we want these five to exist.
    required = {"self_intro", "motivation", "experience", "weakness", "technical"}
    assert required.issubset(categories), f"missing: {required - categories}"
```

- [ ] **Step 2: Run the test — expect failure (fixture does not exist)**

```bash
python -m pytest scripts/tests/test_korean_prompts_fixture.py -v
```

Expected: first test fails with `Missing fixture`.

- [ ] **Step 3: Create the fixture directory and JSON**

```bash
mkdir -p scripts/fixtures
```

Create `scripts/fixtures/korean_prompts.json`:

```json
{
  "description": "Gate #1 Korean interview prompts — used to A/B test Gemma 4 E2B vs Gemini 2.0 Flash.",
  "language": "ko-KR",
  "register": "formal",
  "prompts": [
    {
      "id": "p01",
      "category": "self_intro",
      "text": "안녕하세요. 먼저 간단히 자기소개 부탁드립니다."
    },
    {
      "id": "p02",
      "category": "motivation",
      "text": "저희 회사에 지원하신 이유를 말씀해 주세요."
    },
    {
      "id": "p03",
      "category": "experience",
      "text": "최근에 진행하신 프로젝트 중 가장 기억에 남는 것과 본인의 역할을 설명해 주세요."
    },
    {
      "id": "p04",
      "category": "weakness",
      "text": "본인의 단점과, 그것을 개선하기 위해 어떤 노력을 하고 계신지 말씀해 주세요."
    },
    {
      "id": "p05",
      "category": "strength",
      "text": "본인이 가진 가장 큰 강점은 무엇이라고 생각하시나요? 사례와 함께 답변 부탁드립니다."
    },
    {
      "id": "p06",
      "category": "technical",
      "text": "실시간 멀티모달 AI 시스템에서 레이턴시를 줄이기 위해 어떤 기술적 선택을 하실 건가요?"
    },
    {
      "id": "p07",
      "category": "teamwork",
      "text": "팀원과 의견이 달랐던 경험을 말씀해 주시고, 어떻게 해결하셨는지 알려 주세요."
    },
    {
      "id": "p08",
      "category": "failure",
      "text": "크게 실패했던 경험과 그로부터 배운 점을 한 가지만 말씀해 주세요."
    },
    {
      "id": "p09",
      "category": "growth",
      "text": "최근 1년간 가장 많이 성장했다고 느끼는 부분은 무엇인가요?"
    },
    {
      "id": "p10",
      "category": "closing",
      "text": "마지막으로 저희에게 궁금한 점이 있으시면 편하게 질문해 주세요."
    }
  ]
}
```

- [ ] **Step 4: Run tests and verify all pass**

```bash
python -m pytest scripts/tests/test_korean_prompts_fixture.py -v
```

Expected: 6 tests pass.

- [ ] **Step 5: Commit fixture + structural test**

```bash
git add scripts/fixtures/korean_prompts.json scripts/tests/test_korean_prompts_fixture.py
git commit -m "test(m0): korean_prompts fixture (10 interview prompts) + structural test

Categories cover: self_intro, motivation, experience, weakness, strength,
technical, teamwork, failure, growth, closing. Formal register (존댓말).
Structural test guards against silent edits."
```

---

## Task 4: Gate #1 — Gemma Korean Dialogue Quality Script

**Files:**
- Create: `scripts/smoke_gemma_korean.py`

Interactive CLI: loads the 10 prompts, queries both Gemma 4 E2B and Gemini 2.0 Flash, shows responses side-by-side (blind labels A/B), prompts operator for ratings, writes JSON results.

- [ ] **Step 1: Write the Gate #1 script**

Create `scripts/smoke_gemma_korean.py`:

```python
"""
smoke_gemma_korean.py — Gate #1 from design spec §8.1.

Loads 10 Korean interview prompts. For each:
  - Runs Gemma 4 E2B locally (LiteRT-LM)
  - Runs Gemini 2.0 Flash (cloud) as baseline
  - Shows both responses side-by-side with blind labels (A / B)
  - Prompts operator for ratings (fluency/relevance/register, 1-5)
Aggregates and writes results JSON + prints pass/fail verdict.

PASS criteria (spec §8.1 Gate #1):
  - mean rating across all axes and prompts >= 3.5 / 5
  - no individual rating <= 2.0
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

# Rich imports are used only for terminal UX, not for correctness.
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt

from scripts._smoke_helpers import Rating, aggregate_ratings, format_pass_fail

REPO_ROOT = Path(__file__).parent.parent
MODEL_PATH = REPO_ROOT / "models" / "gemma-4-e2b" / "gemma-4-E2B-it.litertlm"
FIXTURE = REPO_ROOT / "scripts" / "fixtures" / "korean_prompts.json"
RESULTS_DIR = REPO_ROOT / "scripts" / "_results"

# Pass/fail thresholds (spec §8.1 Gate #1)
MIN_MEAN = 3.5
MIN_INDIVIDUAL = 2.0  # any rating <= this is a fail

# System message appended to each prompt (mirrors production registry).
SYSTEM_MSG = (
    "당신은 한국어 면접관입니다. "
    "정중한 존댓말로, 2-3문장 이내로 간결하게 응답하세요. "
    "면접 주제에서 벗어나지 마세요."
)


def _load_gemma() -> Callable[[str], str]:
    from litert_lm import LlmInference  # type: ignore

    engine = LlmInference.create_from_file(str(MODEL_PATH))

    def gen(user_msg: str) -> str:
        full_prompt = f"{SYSTEM_MSG}\n\n사용자: {user_msg}\n면접관:"
        return engine.generate_response(full_prompt)

    return gen


def _load_gemini() -> Callable[[str], str]:
    # Reuse the project's existing Gemini wrapper for fair comparison.
    # This is imported lazily so unit tests don't need google-generativeai.
    sys.path.insert(0, str(REPO_ROOT))
    from start_app.dialogue_manager.meeting import gemini_chat_call  # type: ignore

    def gen(user_msg: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ]
        return gemini_chat_call(messages, max_tokens=200, temperature=0.7)

    return gen


def _collect_rating(console: Console, prompt_id: str) -> Rating:
    console.print(f"[bold]Rate prompt [{prompt_id}][/bold] on three axes (1..5):")
    fluency = IntPrompt.ask("  유창성 (Korean fluency)", choices=["1", "2", "3", "4", "5"])
    relevance = IntPrompt.ask("  관련성 (on-topic)", choices=["1", "2", "3", "4", "5"])
    register = IntPrompt.ask("  격식 (formal register)", choices=["1", "2", "3", "4", "5"])
    return Rating(fluency=fluency, relevance=relevance, register=register)


def main() -> int:
    console = Console()
    RESULTS_DIR.mkdir(exist_ok=True)

    console.print(Panel.fit("[bold]Gate #1 — Gemma 4 E2B vs Gemini 2.0 Flash (Korean)[/bold]"))
    prompts = json.loads(FIXTURE.read_text(encoding="utf-8"))["prompts"]

    console.print("Loading models...")
    gemma = _load_gemma()
    gemini = _load_gemini()
    console.print("[green]Both models ready.[/green]\n")

    per_prompt: list[dict] = []
    ratings_gemma: list[Rating] = []
    ratings_gemini: list[Rating] = []

    for p in prompts:
        console.rule(f"[bold]{p['id']}[/bold] — {p['category']}")
        console.print(f"[italic]Prompt:[/italic] {p['text']}\n")

        # Blind assignment — randomise which model is shown as A vs B per prompt.
        show_gemma_first = random.random() < 0.5
        resp_gemma = gemma(p["text"])
        resp_gemini = gemini(p["text"])

        label_a, label_b = ("gemma", "gemini") if show_gemma_first else ("gemini", "gemma")
        resp_a = resp_gemma if show_gemma_first else resp_gemini
        resp_b = resp_gemini if show_gemma_first else resp_gemma

        console.print(Panel(resp_a, title="[bold]Response A[/bold]", expand=False))
        console.print(Panel(resp_b, title="[bold]Response B[/bold]", expand=False))

        console.print("\n[cyan]Rate response A:[/cyan]")
        r_a = _collect_rating(console, f"{p['id']}-A")
        console.print("\n[cyan]Rate response B:[/cyan]")
        r_b = _collect_rating(console, f"{p['id']}-B")

        # Map blind ratings back to the actual model.
        r_gemma = r_a if label_a == "gemma" else r_b
        r_gemini = r_b if label_b == "gemini" else r_a
        ratings_gemma.append(r_gemma)
        ratings_gemini.append(r_gemini)

        per_prompt.append({
            "id": p["id"],
            "category": p["category"],
            "prompt_text": p["text"],
            "gemma_response": resp_gemma,
            "gemini_response": resp_gemini,
            "gemma_rating": r_gemma.__dict__,
            "gemini_rating": r_gemini.__dict__,
        })

    agg_gemma = aggregate_ratings(ratings_gemma)
    agg_gemini = aggregate_ratings(ratings_gemini)

    passed = agg_gemma["mean"] >= MIN_MEAN and agg_gemma["min"] > MIN_INDIVIDUAL

    result = {
        "gate": "1_korean_dialogue_quality",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "thresholds": {"min_mean": MIN_MEAN, "min_individual_floor": MIN_INDIVIDUAL},
        "gemma": agg_gemma,
        "gemini_baseline": agg_gemini,
        "passed": passed,
        "per_prompt": per_prompt,
    }
    out_path = RESULTS_DIR / "gate1_korean.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    console.print("\n" + "=" * 60)
    console.print(f"[bold]Gate #1 result:[/bold] {format_pass_fail(passed)}")
    console.print(f"  Gemma 4 E2B mean = {agg_gemma['mean']:.2f} (threshold ≥ {MIN_MEAN})")
    console.print(f"  Gemma 4 E2B min  = {agg_gemma['min']} (threshold > {MIN_INDIVIDUAL})")
    console.print(f"  Gemini baseline  mean = {agg_gemini['mean']:.2f}")
    console.print(f"  Results written to: {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    random.seed(42)  # reproducible blind assignment
    sys.exit(main())
```

- [ ] **Step 2: Add `_results/` to .gitignore**

The per-prompt JSON contains full model outputs and is operator-specific; keep it out of git.

```bash
echo "scripts/_results/" >> .gitignore
```

- [ ] **Step 3: Syntax check the script**

```bash
source .venv-smoke/bin/activate
python -c "import ast; ast.parse(open('scripts/smoke_gemma_korean.py').read()); print('syntax OK')"
```

Expected: `syntax OK`.

- [ ] **Step 4: Run Gate #1 end-to-end (manual)**

```bash
# Ensure GOOGLE_API_KEY is set for the Gemini baseline
python -m scripts.smoke_gemma_korean
```

Operator action: rate each response blind. Takes ~15-25 minutes.

At the end, the script prints pass/fail and writes `scripts/_results/gate1_korean.json`.

- [ ] **Step 5: Commit the Gate #1 script**

```bash
git add scripts/smoke_gemma_korean.py .gitignore
git commit -m "feat(m0): Gate #1 — Gemma KR dialogue quality smoke script

Blind A/B comparison against Gemini 2.0 Flash baseline across 10 formal
Korean interview prompts. Operator rates fluency/relevance/register.

PASS criteria (spec §8.1 Gate #1):
  - mean ≥ 3.5/5 across all axes and prompts
  - no individual rating ≤ 2

Writes scripts/_results/gate1_korean.json (gitignored)."
```

---

## Task 5: Multimodal Fixtures (Audio + Photo)

**Files:**
- Create: `scripts/fixtures/audio_samples/self_intro.wav` (and 4 more)
- Create: `scripts/fixtures/test_photo.jpg`
- Create: `scripts/fixtures/expected_emotions.json`

Recorded by the operator on the same M2 MacBook. These are committed with Git LFS if your repo uses it, otherwise stored locally and ignored (Gate #2 is re-runnable per machine; a fresh recording is fine).

- [ ] **Step 1: Record 5 Korean audio samples on the operator's machine**

Use the macOS `음성 메모` (Voice Memos) or QuickTime Player. Each sample is 10-20 seconds, 16kHz mono WAV (or re-encode).

Sample content (the operator speaks these):

1. `self_intro.wav` — "안녕하세요, 저는 김진우라고 합니다. 컴퓨터공학을 전공하고 있습니다."
2. `motivation.wav` — "이 포지션에 지원한 이유는 실시간 AI 시스템에 관심이 많기 때문입니다."
3. `experience.wav` — "최근에 저는 멀티모달 면접 플랫폼을 개발하고 있습니다."
4. `weakness.wav` — "제 단점은 한 가지 일에 너무 몰입해서 다른 일을 놓치는 경우가 있다는 점입니다."
5. `question.wav` — "혹시 팀의 엔지니어링 문화에 대해 조금 더 알려주실 수 있을까요?"

Convert to 16kHz mono WAV (Gemma audio input spec):

```bash
mkdir -p scripts/fixtures/audio_samples
# From a recorded m4a or wav input:
for NAME in self_intro motivation experience weakness question; do
  ffmpeg -y -i ~/path/to/recording-${NAME}.m4a \
    -ar 16000 -ac 1 \
    scripts/fixtures/audio_samples/${NAME}.wav
done
```

Each resulting file should be ~300-700 KB.

- [ ] **Step 2: Take a neutral test photo on the operator's webcam**

Use Photo Booth or any camera app. Face visible, neutral/calm expression, good lighting.

Save (and resize to a modest size) as `scripts/fixtures/test_photo.jpg`:

```bash
# If your source is large, resize to ~512x384 at quality 80 to mirror production frames:
sips -Z 512 ~/path/to/source_photo.jpg --out scripts/fixtures/test_photo.jpg
```

Target file size: 30-80 KB.

- [ ] **Step 3: Create the expected-emotions fixture**

Create `scripts/fixtures/expected_emotions.json`:

```json
{
  "description": "Ground truth for Gate #2 emotion-inference check. Manually labelled.",
  "schema_note": "Each audio sample is paired with the same test_photo.jpg (neutral). Expected emotion reflects the tone in the audio, not the photo. Confidence and engagement on the same high/medium/low scale as production emotion JSON (spec §4.6).",
  "samples": [
    {
      "audio": "self_intro.wav",
      "expected_confidence": "medium",
      "expected_engagement": "medium",
      "expected_transcript": "안녕하세요, 저는 김진우라고 합니다. 컴퓨터공학을 전공하고 있습니다."
    },
    {
      "audio": "motivation.wav",
      "expected_confidence": "high",
      "expected_engagement": "high",
      "expected_transcript": "이 포지션에 지원한 이유는 실시간 AI 시스템에 관심이 많기 때문입니다."
    },
    {
      "audio": "experience.wav",
      "expected_confidence": "medium",
      "expected_engagement": "high",
      "expected_transcript": "최근에 저는 멀티모달 면접 플랫폼을 개발하고 있습니다."
    },
    {
      "audio": "weakness.wav",
      "expected_confidence": "medium",
      "expected_engagement": "medium",
      "expected_transcript": "제 단점은 한 가지 일에 너무 몰입해서 다른 일을 놓치는 경우가 있다는 점입니다."
    },
    {
      "audio": "question.wav",
      "expected_confidence": "medium",
      "expected_engagement": "high",
      "expected_transcript": "혹시 팀의 엔지니어링 문화에 대해 조금 더 알려주실 수 있을까요?"
    }
  ]
}
```

- [ ] **Step 4: Commit the fixtures**

Audio samples are small (< 1 MB each), so commit directly without LFS.

```bash
git add scripts/fixtures/expected_emotions.json \
        scripts/fixtures/audio_samples/ \
        scripts/fixtures/test_photo.jpg
git commit -m "test(m0): multimodal fixtures (5 KR audio + 1 photo + labels)

- 5 x 16 kHz mono WAV samples matching production audio format (spec §5.2)
- 1 neutral test_photo.jpg, 512 px long edge, JPEG quality ~80
- expected_emotions.json: ground-truth transcripts + confidence/engagement
  labels (high/medium/low) matching production schema (spec §4.6)"
```

---

## Task 6: Gate #2 — Gemma Multimodal (Audio + Vision) Script

**Files:**
- Create: `scripts/smoke_gemma_multimodal.py`

Runs Gemma 4 E2B with audio + image input. Verifies (a) transcription WER vs. labelled ground truth and (b) emotion JSON matches labels.

- [ ] **Step 1: Write the Gate #2 script**

Create `scripts/smoke_gemma_multimodal.py`:

```python
"""
smoke_gemma_multimodal.py — Gate #2 from design spec §8.1.

For each of 5 recorded Korean audio samples + a shared test photo:
  - Invoke Gemma 4 E2B with audio_bytes + image_bytes
  - Ask it to transcribe + produce emotion JSON (matching production schema)
  - Compare transcript WER to expected_transcript
  - Compare emotion confidence+engagement to expected labels

PASS criteria (spec §8.1 Gate #2):
  - mean WER across 5 samples <= 0.10 (≥90% transcription accuracy)
  - emotion (confidence + engagement) exact match on ≥ 3 of 5 samples
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from scripts._smoke_helpers import format_pass_fail, word_error_rate

REPO_ROOT = Path(__file__).parent.parent
MODEL_PATH = REPO_ROOT / "models" / "gemma-4-e2b" / "gemma-4-E2B-it.litertlm"
AUDIO_DIR = REPO_ROOT / "scripts" / "fixtures" / "audio_samples"
PHOTO = REPO_ROOT / "scripts" / "fixtures" / "test_photo.jpg"
LABELS = REPO_ROOT / "scripts" / "fixtures" / "expected_emotions.json"
RESULTS_DIR = REPO_ROOT / "scripts" / "_results"

# Pass/fail thresholds
MAX_MEAN_WER = 0.10
MIN_EMOTION_MATCHES = 3  # out of 5

PROMPT = (
    "첨부된 오디오와 이미지를 분석하세요. "
    "응답은 반드시 JSON 한 개로만 출력하세요. 추가 설명 금지. "
    "스키마: {\"transcript\": \"오디오 전사 한국어\", "
    '"confidence": "high|medium|low", '
    '"engagement": "high|medium|low", '
    '"note": "간단한 관찰 60자 이내"}'
)


def _load_multimodal_engine():
    """Return callable that takes (audio_bytes, image_bytes) → str (JSON response).

    Uses LiteRT-LM multimodal API. If your installed version exposes a different
    method (e.g. `.generate_multimodal` vs `.generate_response_multi`), update
    the single call below — the rest of the script is stable.
    """
    from litert_lm import LlmInference  # type: ignore

    engine = LlmInference.create_from_file(str(MODEL_PATH))

    def gen(audio_bytes: bytes, image_bytes: bytes) -> str:
        return engine.generate_multimodal(
            prompt=PROMPT,
            audio=audio_bytes,
            image=image_bytes,
            max_output_tokens=300,
        )

    return gen


def _parse_emotion(raw: str) -> dict[str, Any]:
    """Extract the JSON object from Gemma's response. Lenient to leading text."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {"transcript": "", "confidence": "unknown", "engagement": "unknown", "note": raw[:80]}
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {"transcript": "", "confidence": "unknown", "engagement": "unknown", "note": raw[:80]}


def main() -> int:
    console = Console()
    RESULTS_DIR.mkdir(exist_ok=True)
    console.rule("[bold]Gate #2 — Gemma 4 E2B multimodal (audio + vision)[/bold]")

    console.print("Loading model...")
    generate = _load_multimodal_engine()
    console.print("[green]Model ready.[/green]\n")

    labels = json.loads(LABELS.read_text(encoding="utf-8"))["samples"]
    image_bytes = PHOTO.read_bytes()

    per_sample: list[dict] = []
    wer_values: list[float] = []
    emotion_matches = 0

    table = Table(title="Gate #2 per-sample results")
    table.add_column("Sample")
    table.add_column("WER")
    table.add_column("Confidence (exp / got)")
    table.add_column("Engagement (exp / got)")

    for sample in labels:
        audio_path = AUDIO_DIR / sample["audio"]
        audio_bytes = audio_path.read_bytes()
        raw = generate(audio_bytes, image_bytes)
        parsed = _parse_emotion(raw)

        wer = word_error_rate(
            reference=sample["expected_transcript"],
            hypothesis=parsed.get("transcript", ""),
        )
        wer_values.append(wer)

        conf_match = parsed.get("confidence") == sample["expected_confidence"]
        eng_match = parsed.get("engagement") == sample["expected_engagement"]
        if conf_match and eng_match:
            emotion_matches += 1

        per_sample.append({
            "audio": sample["audio"],
            "expected_transcript": sample["expected_transcript"],
            "got_transcript": parsed.get("transcript", ""),
            "wer": wer,
            "expected_confidence": sample["expected_confidence"],
            "got_confidence": parsed.get("confidence"),
            "expected_engagement": sample["expected_engagement"],
            "got_engagement": parsed.get("engagement"),
            "raw_response": raw,
        })

        table.add_row(
            sample["audio"],
            f"{wer:.2%}",
            f"{sample['expected_confidence']} / {parsed.get('confidence')}",
            f"{sample['expected_engagement']} / {parsed.get('engagement')}",
        )

    console.print(table)

    mean_wer = sum(wer_values) / len(wer_values) if wer_values else 1.0
    wer_pass = mean_wer <= MAX_MEAN_WER
    emotion_pass = emotion_matches >= MIN_EMOTION_MATCHES
    passed = wer_pass and emotion_pass

    result = {
        "gate": "2_multimodal",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "thresholds": {"max_mean_wer": MAX_MEAN_WER, "min_emotion_matches": MIN_EMOTION_MATCHES},
        "mean_wer": mean_wer,
        "emotion_matches": emotion_matches,
        "wer_pass": wer_pass,
        "emotion_pass": emotion_pass,
        "passed": passed,
        "per_sample": per_sample,
    }
    out_path = RESULTS_DIR / "gate2_multimodal.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    console.print("\n" + "=" * 60)
    console.print(f"[bold]Gate #2 result:[/bold] {format_pass_fail(passed)}")
    console.print(f"  Mean WER        = {mean_wer:.2%} (threshold ≤ {MAX_MEAN_WER:.0%})")
    console.print(f"  Emotion matches = {emotion_matches}/5 (threshold ≥ {MIN_EMOTION_MATCHES}/5)")
    console.print(f"  Results written to: {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Syntax check**

```bash
python -c "import ast; ast.parse(open('scripts/smoke_gemma_multimodal.py').read()); print('syntax OK')"
```

Expected: `syntax OK`.

- [ ] **Step 3: Run Gate #2 end-to-end (manual)**

```bash
source .venv-smoke/bin/activate
python -m scripts.smoke_gemma_multimodal
```

Runs unattended. Takes ~1-3 minutes depending on Gemma inference speed. Writes `scripts/_results/gate2_multimodal.json`.

- [ ] **Step 4: Commit Gate #2 script**

```bash
git add scripts/smoke_gemma_multimodal.py
git commit -m "feat(m0): Gate #2 — Gemma multimodal (audio+vision) smoke script

For each of 5 audio samples + 1 photo:
  - Invoke Gemma with audio_bytes + image_bytes → JSON response
  - Compute transcript WER vs labelled ground truth
  - Compare emotion labels (confidence+engagement)

PASS criteria (spec §8.1 Gate #2):
  - mean WER ≤ 10%
  - exact emotion match on ≥ 3 of 5 samples

Writes scripts/_results/gate2_multimodal.json."
```

---

## Task 7: Gate #3 — M2 Latency Benchmark Script

**Files:**
- Create: `scripts/bench_latency.py`

Measures end-to-end latency on M2 hardware for a representative turn: audio buffer + frame → Gemma first token → sentence boundary → ElevenLabs first byte. Reports p50/p95/p99 over 20 iterations.

- [ ] **Step 1: Write the benchmark script**

Create `scripts/bench_latency.py`:

```python
"""
bench_latency.py — Gate #3 from design spec §8.1.

Measures end-to-end latency on the host M2 MacBook:
  1. Gemma 4 E2B inference (audio + frame → first sentence)
  2. Simulated sentence-boundary detection
  3. ElevenLabs streaming TTS first-byte (real API call)
  4. Total = Gemma-to-first-sentence + ElevenLabs-first-byte

Runs 20 iterations with a 3-turn rolling history (matches production dialog
depth). Reports p50/p95/p99.

PASS criteria (spec §8.1 Gate #3):
  - p50 < 2.5s
  - p95 < 4.0s
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from scripts._smoke_helpers import compute_percentiles, format_pass_fail

REPO_ROOT = Path(__file__).parent.parent
MODEL_PATH = REPO_ROOT / "models" / "gemma-4-e2b" / "gemma-4-E2B-it.litertlm"
AUDIO = REPO_ROOT / "scripts" / "fixtures" / "audio_samples" / "self_intro.wav"
PHOTO = REPO_ROOT / "scripts" / "fixtures" / "test_photo.jpg"
RESULTS_DIR = REPO_ROOT / "scripts" / "_results"

ITERATIONS = 20
PASS_P50_S = 2.5
PASS_P95_S = 4.0

FIRST_SENTENCE_END = ".!?。?!\n"


def _gemma_first_sentence(engine, audio_bytes: bytes, image_bytes: bytes, history: list[dict]) -> tuple[str, float]:
    """Stream Gemma tokens until a sentence-ender is emitted. Return (sentence, seconds)."""
    t0 = time.perf_counter()
    buf = []
    hist_preamble = "\n".join(f"[{t['role']}] {t['text']}" for t in history)
    prompt = (
        "이전 대화:\n" + hist_preamble + "\n\n" +
        "사용자의 오디오와 이미지를 듣고 보고 한국어로 간결하게 답변하세요."
    )
    # Streaming API — replace `stream_multimodal` with the actual method your
    # installed litert-lm exposes (common names: generate_multimodal_stream,
    # stream_generate). Structure is stable.
    for token in engine.stream_multimodal(prompt=prompt, audio=audio_bytes, image=image_bytes):
        buf.append(token)
        joined = "".join(buf)
        if joined and joined[-1] in FIRST_SENTENCE_END:
            return joined, time.perf_counter() - t0
        if len(buf) > 40:
            return joined, time.perf_counter() - t0
    return "".join(buf), time.perf_counter() - t0


def _elevenlabs_first_byte(text: str) -> float:
    """Call ElevenLabs streaming TTS, return seconds until first audio byte."""
    from elevenlabs.client import ElevenLabs  # type: ignore

    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    t0 = time.perf_counter()
    stream = client.text_to_speech.convert_as_stream(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    # First byte = first chunk yielded.
    for _ in stream:
        return time.perf_counter() - t0
    return time.perf_counter() - t0


def main() -> int:
    console = Console()
    RESULTS_DIR.mkdir(exist_ok=True)
    console.rule("[bold]Gate #3 — M2 MacBook latency benchmark[/bold]")

    console.print("Loading Gemma...")
    from litert_lm import LlmInference  # type: ignore
    engine = LlmInference.create_from_file(str(MODEL_PATH))

    audio_bytes = AUDIO.read_bytes()
    image_bytes = PHOTO.read_bytes()

    # Warmup (first inference is always slower; excluded from stats).
    console.print("Warmup iteration (not counted)...")
    _gemma_first_sentence(engine, audio_bytes, image_bytes, history=[])

    history = [
        {"role": "assistant", "text": "안녕하세요, 자기소개 부탁드립니다."},
        {"role": "user", "text": "안녕하세요. 저는 김진우입니다."},
        {"role": "assistant", "text": "네, 반갑습니다. 지원 동기를 말씀해 주세요."},
    ]

    total_latencies: list[float] = []
    gemma_latencies: list[float] = []
    tts_latencies: list[float] = []

    for i in range(ITERATIONS):
        sentence, gemma_s = _gemma_first_sentence(engine, audio_bytes, image_bytes, history)
        tts_s = _elevenlabs_first_byte(sentence or "안녕하세요.")
        total = gemma_s + tts_s
        gemma_latencies.append(gemma_s)
        tts_latencies.append(tts_s)
        total_latencies.append(total)
        console.print(f"  [{i+1:2d}/{ITERATIONS}] gemma={gemma_s:.2f}s  tts={tts_s:.2f}s  total={total:.2f}s")

    g = compute_percentiles(gemma_latencies)
    t = compute_percentiles(tts_latencies)
    tot = compute_percentiles(total_latencies)

    passed = tot["p50"] < PASS_P50_S and tot["p95"] < PASS_P95_S

    table = Table(title="Latency summary (seconds)")
    table.add_column("Stage")
    table.add_column("p50")
    table.add_column("p95")
    table.add_column("p99")
    table.add_row("Gemma first-sentence", f"{g['p50']:.2f}", f"{g['p95']:.2f}", f"{g['p99']:.2f}")
    table.add_row("ElevenLabs first-byte", f"{t['p50']:.2f}", f"{t['p95']:.2f}", f"{t['p99']:.2f}")
    table.add_row("[bold]TOTAL[/bold]", f"{tot['p50']:.2f}", f"{tot['p95']:.2f}", f"{tot['p99']:.2f}")
    console.print(table)

    result = {
        "gate": "3_latency",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "iterations": ITERATIONS,
        "thresholds": {"p50_max_s": PASS_P50_S, "p95_max_s": PASS_P95_S},
        "gemma": g,
        "elevenlabs": t,
        "total": tot,
        "passed": passed,
        "raw_totals_s": total_latencies,
    }
    out_path = RESULTS_DIR / "gate3_latency.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    console.print("\n" + "=" * 60)
    console.print(f"[bold]Gate #3 result:[/bold] {format_pass_fail(passed)}")
    console.print(f"  Total p50 = {tot['p50']:.2f}s (threshold < {PASS_P50_S})")
    console.print(f"  Total p95 = {tot['p95']:.2f}s (threshold < {PASS_P95_S})")
    console.print(f"  Results written to: {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Syntax check**

```bash
python -c "import ast; ast.parse(open('scripts/bench_latency.py').read()); print('syntax OK')"
```

Expected: `syntax OK`.

- [ ] **Step 3: Run Gate #3 (requires ELEVENLABS_API_KEY)**

```bash
source .venv-smoke/bin/activate
export ELEVENLABS_API_KEY="$(grep ELEVENLABS_API_KEY start_app/dialogue_manager/keys.py | head -1 | cut -d'"' -f2)"
python -m scripts.bench_latency
```

Runs 20 iterations (~3-7 minutes). Writes `scripts/_results/gate3_latency.json`.

- [ ] **Step 4: Commit Gate #3**

```bash
git add scripts/bench_latency.py
git commit -m "feat(m0): Gate #3 — M2 latency benchmark

Per-turn: Gemma streaming first-sentence + ElevenLabs TTS first-byte.
20 iterations with 3-turn rolling history to match production dialog depth.

PASS criteria (spec §8.1 Gate #3): p50 < 2.5s, p95 < 4.0s."
```

---

## Task 8: Consolidated Decision Document

**Files:**
- Create: `scripts/smoke_results.md`
- Modify: `README.md` (add Smoke Tests section)

This is the GO/NO-GO artifact that lives in git as the formal project record.

- [ ] **Step 1: After all three gates have been run locally, write the decision document**

Create `scripts/smoke_results.md` with the actual numbers from your `_results/*.json` runs:

```markdown
# Parlor-hybrid Migration — M0 Smoke Gate Results

**Run date:** <YYYY-MM-DD>
**Machine:** MacBook Pro / Air M2, macOS <version>, <RAM> GB
**Reviewer:** <your name>
**Design spec:** `docs/superpowers/specs/2026-04-17-parlor-hybrid-migration-design.md` §8.1

---

## Summary

| Gate | Outcome | Key number |
|------|---------|-----------|
| #1 Korean dialogue quality | ✅ PASS / ❌ FAIL | Gemma mean: <X.XX> / 5 (threshold ≥ 3.5) |
| #2 Multimodal (audio+vision) | ✅ PASS / ❌ FAIL | Mean WER: <X.XX%> ; Emotion match: <N>/5 |
| #3 M2 latency | ✅ PASS / ❌ FAIL | Total p50/p95: <X.XX>/<X.XX>s |

**Decision:** 🟢 **GO** — proceed to M1 / 🔴 **NO-GO** — halt migration (see §2.3 of spec)

---

## Gate #1 — Korean Dialogue Quality

Full JSON: `scripts/_results/gate1_korean.json` (local only, gitignored)

- Gemma 4 E2B mean: <value>
- Gemma 4 E2B min individual rating: <value>
- Gemini 2.0 Flash baseline mean: <value>
- Most notable Gemma failure mode (if any): <short description>

## Gate #2 — Multimodal

Full JSON: `scripts/_results/gate2_multimodal.json`

- Mean WER: <value>
- Emotion exact matches: <N>/5
- Worst transcription: `audio_samples/<file>` — WER <value>

## Gate #3 — M2 Latency

Full JSON: `scripts/_results/gate3_latency.json`

- Gemma first-sentence p50/p95: <X>/<Y>s
- ElevenLabs first-byte p50/p95: <X>/<Y>s
- Total p50/p95/p99: <X>/<Y>/<Z>s

## Notes & Follow-ups

- Any observations worth carrying forward to M1+
- Any open question from spec §10 that is now answered
- If NO-GO: recommended pivot (e.g., keep Gemini cloud LLM in hybrid) or project halt
```

- [ ] **Step 2: Add a README section pointing at the smoke tests**

Append to `README.md` (or create a new "Smoke Tests" section):

```markdown

## Realtime Mode Smoke Tests (M0 Validation)

Before proceeding with the Parlor-hybrid migration (`start_app_ws/`), run the
three GO/NO-GO gates described in
`docs/superpowers/specs/2026-04-17-parlor-hybrid-migration-design.md` §8.1:

```bash
python3.12 -m venv .venv-smoke
source .venv-smoke/bin/activate
pip install -r requirements-smoke.txt
# Download Gemma 4 E2B .litertlm (see plan Task 1 Step 3), place in models/gemma-4-e2b/

python -m scripts.smoke_gemma_korean     # Gate #1 (interactive, ~20 min)
python -m scripts.smoke_gemma_multimodal # Gate #2 (unattended, ~2 min)
python -m scripts.bench_latency          # Gate #3 (unattended, ~5 min)
```

Record the outcome in `scripts/smoke_results.md` and commit.
```

- [ ] **Step 3: Commit the results document and README**

```bash
git add scripts/smoke_results.md README.md
git commit -m "docs(m0): smoke_results.md + README Smoke Tests section

Formalises the GO/NO-GO decision produced by running all three gates.
Template filled in with actual numbers once gates have been run locally."
```

---

## Task 9: Final PR

- [ ] **Step 1: Push the feature branch**

```bash
git push origin parlor-hybrid/m0-smoke-gates
```

- [ ] **Step 2: Open the PR**

```bash
~/bin/gh pr create --fill --base main --head parlor-hybrid/m0-smoke-gates \
  --title "[M0] Pre-migration Smoke Gates" \
  --body "$(cat <<'EOF'
## Summary

Implements M0 of the Parlor-hybrid migration (design spec §8.1 / §9.2).

- Gate #1: Gemma 4 E2B vs Gemini 2.0 Flash Korean dialogue quality (blind A/B rating)
- Gate #2: Gemma 4 E2B multimodal (audio+vision) — WER + emotion label match
- Gate #3: M2 MacBook end-to-end latency (Gemma + ElevenLabs first-byte, 20 iters)
- Shared helpers with 12 unit tests
- Smoke-only isolated venv (`requirements-smoke.txt`) — production env untouched

## Scope boundary

This PR adds scripts/fixtures/helpers only. **Zero changes to `start_app/` or production code.** The Legacy Gemini-cloud path continues to run unchanged.

## Result

See `scripts/smoke_results.md` for the GO/NO-GO decision. If NO-GO, the next PR will be a pivot plan, not M1.

## Test plan

- [x] `pytest scripts/tests/` — 18 passing (helpers + fixture structure)
- [x] `python -m scripts.smoke_gemma_korean` — Gate #1 executed
- [x] `python -m scripts.smoke_gemma_multimodal` — Gate #2 executed
- [x] `python -m scripts.bench_latency` — Gate #3 executed
- [x] `smoke_results.md` filled in with actual numbers

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Create GitHub issue #16 (if not already) and link it**

```bash
# If issue #16 does not yet exist in the repo, create it to track the milestone:
~/bin/gh issue create \
  --title "[M0] Pre-migration Smoke Gates (Gemma KR + multimodal + M2 latency)" \
  --body "Validate GO/NO-GO for Parlor-hybrid migration. See docs/superpowers/specs/2026-04-17-parlor-hybrid-migration-design.md §8.1 and plan docs/superpowers/plans/2026-04-17-parlor-hybrid-m0-smoke-gates.md."
```

Link the PR to the issue (GitHub will auto-link if the PR body mentions `#16`).

---

## Self-Review

### Spec coverage

- §8.1 Gate #1 — Task 4 (`smoke_gemma_korean.py`) ✅
- §8.1 Gate #2 — Task 5 (fixtures) + Task 6 (`smoke_gemma_multimodal.py`) ✅
- §8.1 Gate #3 — Task 7 (`bench_latency.py`) ✅
- §8.2 unit tests — Task 2 (`test_smoke_helpers.py`) ✅ — scoped to helpers only, which is correct for M0
- §9.1 Step 5-1 cleanup — Prerequisites P0 ✅
- §9.2 M0 "Done criteria: All 3 gates pass; GO decision documented" — Task 8 (`smoke_results.md`) ✅
- §9.3 branching `parlor-hybrid/m0-smoke-gates` — Prerequisites P0 Step 5 ✅
- §9.5 GitHub issue mapping — Task 9 Step 3 ✅

### Placeholder scan

- No "TBD", "TODO", "implement later" markers remain in the plan.
- LiteRT-LM API method names (`create_from_file`, `generate_response`, `generate_multimodal`, `stream_multimodal`) are best-effort based on Google AI Edge docs; each usage flags this with an inline note telling the engineer to adjust to their installed version. The structure around the calls is complete and stable.
- Korean prompt fixtures are concrete text, not placeholders.

### Type consistency

- `Rating` dataclass has fields `fluency`, `relevance`, `register`. These match across Task 2 tests, Task 2 impl, and Task 4 usage.
- `aggregate_ratings` return keys `{"mean", "min", "count"}` — used consistently.
- `compute_percentiles` return keys `{"p50", "p95", "p99", "count"}` — used consistently in Tasks 2, 7.
- Emotion JSON schema `{confidence, engagement, note, transcript}` matches between Task 5 fixtures, Task 6 script parsing, and spec §4.6.
- Results JSON schema is stable: each gate writes a dict with a `passed: bool`, `timestamp`, `thresholds`, and per-sample/per-prompt detail.

### Scope

- M0 is scoped to validation scripts + fixtures + one decision document. No `start_app_ws/`, no browser changes. Matches spec §9.2 "Deliverables" column.
- M1–M7 explicitly out of scope for this plan — separate plans will be written each, once M0 passes.

---

## M1–M7 Outline (future plans, not this one)

If M0 passes, subsequent plans will be written independently. Each is a separate PR against `main`.

| Plan | Est. size | Dependencies |
|------|-----------|--------------|
| `2026-XX-XX-parlor-hybrid-m1-fastapi-skeleton.md` | ~10 tasks | M0 GO |
| `2026-XX-XX-parlor-hybrid-m2-gemma-integration.md` | ~12 tasks | M1 |
| `2026-XX-XX-parlor-hybrid-m3-audio-frame-pipeline.md` | ~14 tasks | M2 |
| `2026-XX-XX-parlor-hybrid-m4-elevenlabs-streaming-tts.md` | ~10 tasks | M3 |
| `2026-XX-XX-parlor-hybrid-m5-vad-bargein.md` | ~8 tasks | M4 |
| `2026-XX-XX-parlor-hybrid-m6-error-fallback.md` | ~10 tasks | M5 |
| `2026-XX-XX-parlor-hybrid-m7-demo-prep.md` | ~6 tasks | M6 |

---
