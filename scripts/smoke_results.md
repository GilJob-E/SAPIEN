# Parlor-hybrid Migration — M0 Smoke Gate Results

**Run date:** 2026-04-17
**Machine:** MacBook M2 base (Apple Silicon), macOS 15.3.0, 16 GB unified memory
**Rater:** Claude controller (automated, per user direction "너가 직접 실행해봐")
**Design spec:** [`docs/superpowers/specs/2026-04-17-parlor-hybrid-migration-design.md`](../docs/superpowers/specs/2026-04-17-parlor-hybrid-migration-design.md) §8.1
**Implementation plan:** [`docs/superpowers/plans/2026-04-17-parlor-hybrid-m0-smoke-gates.md`](../docs/superpowers/plans/2026-04-17-parlor-hybrid-m0-smoke-gates.md)

---

## 1. Summary

| Gate | Strict criterion | Strict result | Substantive result | Decision |
|------|------------------|---------------|--------------------|----------|
| **#1 Korean dialogue quality** | mean≥3.5, min>2 | ❌ FAIL (min=2) | mean=4.30, failures prompt-fixable | ⚠️ **Conditional PASS** |
| **#2 Multimodal transcription** | WER≤10% | ✅ **PASS** (WER=7.71%) | 3/5 samples 0% WER | ✅ **PASS** |
| **#2 Multimodal emotion** | ≥3/5 exact match | ❌ FAIL (0/5) | Methodology issue (TTS audio ≠ varied emotion) | ⚠️ Methodology |
| **#3 Latency (M2 CPU)** | p50<2.5s, p95<4s | ❌ FAIL | p50=4.57s, p95=4.68s | 🚨 **Hard constraint** |

**Decision:** 🔴 **NO-GO on strict spec criteria** — Gate #3 is a hard hardware constraint, not fixable by prompt engineering or label recalibration.

**Recommendation:** Before accepting NO-GO, try **Metal GPU backend** (Gate #3 retry, ~10 min). If GPU backend does not exist or doesn't help enough, pivot to one of:
- **Scope reduction** to hybrid-hybrid (Gemma for STT/Vision only, keep Gemini cloud LLM)
- **Relaxed latency target** (≤4s, accept 9% improvement over baseline for demo)
- **Hardware upgrade path** (M3 Pro / M4 required for 2.5s target)

---

## 2. Gate #1 — Korean Dialogue Quality

**Full data:** [`scripts/_results/gate1_korean.json`](_results/gate1_korean.json), [`scripts/_results/gate1_responses_raw.json`](_results/gate1_responses_raw.json)

**Scope:** 10 Korean interview prompts, 3 axes (fluency, relevance, register), 1-5 scale, rated by Claude controller.

### Aggregated scores

| Model | Mean | Min | Baseline delta |
|-------|------|-----|----------------|
| **Gemma 4 E2B (local)** | **4.30** | **2** | — |
| Gemini 2.5 Flash (cloud baseline) | 4.80 | 3 | +0.50 |

### Failure points (ratings of 2)

| Prompt | Axis | Value | Root cause |
|--------|------|-------|------------|
| p03 experience | Relevance | 2 | Gemma responded with **template placeholders** (`[프로젝트 이름]`, `[본인의 역할]`) instead of concrete content |
| p04 weakness | Fluency | 2 | Gemma **echoed the prompt** and emitted a `**면접관 답변:**` markdown header before answering |

### Mitigations (must land in M2)

Both failures are **prompt-engineering fixable**. The M2 system prompt must include:

```
- 이전 메시지를 절대 인용하거나 반복하지 마세요
- [괄호 플레이스홀더] 사용 금지 — 구체적으로 답하세요
- 마크다운 헤더 금지 — 평문으로만 응답
```

Plus 2-3 few-shot examples showing correct direct answers.

### Production buglet surfaced

The existing `start_app/dialogue_manager/llm.py:29` uses `MODEL = "gemini-2.0-flash"`, which now returns **404 NOT FOUND** for new API keys ("This model is no longer available to new users"). The Legacy fallback path is therefore already broken for fresh credentials. Needs a one-line bump to `gemini-2.5-flash` regardless of migration outcome.

---

## 3. Gate #2 — Multimodal (Audio + Vision)

**Full data:** [`scripts/_results/gate2_multimodal.json`](_results/gate2_multimodal.json)

**Scope:** 5 Korean audio samples (TTS-generated via ElevenLabs for reproducibility) + 1 synthetic neutral photo → Gemma multimodal JSON output with transcript + confidence + engagement.

### Transcription WER (PASS)

| Sample | WER | Difference |
|--------|-----|-----------|
| motivation | **0.00%** | perfect |
| experience | **0.00%** | perfect |
| weakness | **0.00%** | perfect |
| self_intro | 28.6% | `컴퓨터공학` vs `컴퓨터 공학` (spacing — semantically equivalent) |
| question | 10.0% | dropped the filler `혹시` (1 word) |

**Mean WER: 7.71% — PASSES ≤10% threshold.** Three samples are exact matches; the remaining two are semantically equivalent paraphrases. Real transcription capability is excellent.

### Emotion label match (methodology issue, not capability)

| Sample | Expected confidence | Got confidence | Match? |
|--------|--------------------|-----------------|--------|
| self_intro | medium | high | ❌ |
| motivation | medium | high | ❌ |
| experience | medium | high | ❌ |
| weakness | medium | high | ❌ |
| question | medium | high | ❌ |

Gemma labelled **every** sample as `high` confidence. Root cause: the ElevenLabs TTS voice produces clean, professional-sounding audio without hesitation or filler, which Gemma correctly perceives as "high confidence". The labeller (me) used `medium` as a conservative default, not realising TTS audio is uniformly confident.

**This is a labeling-methodology failure, not a Gemma capability failure.** M5 (VAD + real human audio) naturally replaces this with audio that has genuine emotional variance.

---

## 4. Gate #3 — Latency Benchmark (🚨 Hard FAIL)

**Full data:** [`scripts/_results/gate3_latency.json`](_results/gate3_latency.json), [`scripts/_results/multimodal_probe.json`](_results/multimodal_probe.json)

**Scope:** 20 iterations of end-to-end streaming pipeline (Gemma multimodal streaming → sentence boundary → ElevenLabs first audio byte). 3-turn rolling history. M2 MacBook base with `Backend.CPU` for all three backends (main/audio/vision — Metal/GPU support marked "upcoming" in the Python API docs).

### Measured latencies (seconds)

| Stage | p50 | p95 | p99 |
|-------|-----|-----|-----|
| Gemma TTFT (first token out) | 2.93 | 3.11 | 3.40 |
| Gemma first-sentence | 3.10 | 3.29 | 3.58 |
| ElevenLabs first-byte | 1.42 | 1.58 | 1.60 |
| **TOTAL (user-speech-end → AI-audio-start)** | **4.57** | **4.68** | **4.94** |

### Targets vs. actual

| Target (spec §8.1 Gate #3) | Actual | Pass? |
|----------------------------|--------|-------|
| Total p50 < 2.5 s | 4.57 s (**+82%**) | ❌ |
| Total p95 < 4.0 s | 4.68 s (**+17%**) | ❌ |

### Why this fails hard

TTFT alone is 2.93 s. Even with a hypothetical zero-latency TTS, total ≥ 3 s on M2 base CPU. The 2.5 s target is **architecturally unreachable** on this hardware without one of:

1. **Metal GPU backend** (not verified to be available in current `litert-lm-api` Python package for Gemma-4 E2B — needs investigation)
2. **NPU backend** (attempted during env init, registration failed with `kLiteRtStatusErrorInvalidArgument` — not available on M2)
3. **Different hardware** (M3 Pro / M4 class — the machines Parlor's 2.5-3 s benchmark was measured on)
4. **Model downscaling** (smaller Gemma / distilled model — defeats the purpose)

### Comparison to baseline

| Pipeline | E2E |
|----------|-----|
| Current production (cloud, 4 serial RTTs) | ≈5.0 s |
| Parlor-hybrid on M2 CPU (this result) | 4.57 s |
| **Delta** | **-9%** (0.43 s) |

A 9% improvement for a 3-week rewrite + 8 GitHub issues of rollout risk is a poor trade. The migration's primary value proposition (spec §2.3 motivation "Latency ≤ 3 s") is not delivered on this hardware tier.

### The multimodal *pipeline* itself works

The probe ([`multimodal_probe.json`](_results/multimodal_probe.json)) confirmed that the **pipeline restructuring** — single Gemma call replacing 4 serial RTTs — is correct. Upload+inference for 5 s audio + image produces coherent Korean output in 3.75 s steady-state. The "영상/음성 업로드가 오래 걸림" concern (the user's original bottleneck) is genuinely eliminated. What remains is raw compute cost: M2 CPU doesn't have enough FLOPs/s to make 2 B active parameters fast enough.

---

## 5. Open Items / Not Yet Resolved

- [ ] **Metal GPU backend test.** `Backend.GPU` may be exposed for audio/vision/main backends; untested. If TTFT drops to ~1.5 s, total becomes ≈3 s — potential rescue path.
- [ ] **Label methodology for Gate #2 emotion axis.** Re-do with either (a) real recorded audio with varied emotion, (b) different TTS voices for variance, or (c) relaxed criterion (1-axis-out-of-2 match).
- [ ] **Legacy `gemini-2.0-flash` fallback.** Update `start_app/dialogue_manager/llm.py` independent of M0 outcome.
- [ ] **Gate #1 prompt engineering mitigations.** Draft and bank the improved system prompt so it's ready for M2 if migration proceeds.

---

## 6. Recommended Decision Paths

Presented in order of engineering cost / risk:

### Path A — Retry Gate #3 with Metal GPU (10 min)
Low effort. If `Backend.GPU` is accepted and TTFT drops, re-evaluate.

### Path B — Halt migration, ship Step 5-1 only
Honours spec §9.6 rollback ("M0 fail: halt migration, ship Step 5-1 optimizations only"). 9% improvement for 3 weeks of work is not worth the risk. Legacy path with `gemini-2.5-flash` bump and current concise-response prompt is the demo plan.

### Path C — Scope reduction: hybrid-hybrid
Keep Gemma only for STT + Vision (where the upload-latency win lives). Use Gemini cloud for LLM (where it's fastest). This captures the upload-latency improvement without requiring Gemma inference speed to hit target. Estimated total ≈ 2.5-3 s if Gemma STT/Vision in parallel with Gemini LLM.

### Path D — Accept relaxed target
Redefine M0 Gate #3 as `p95 < 5.0 s` (beat current baseline). Ship as-is. Justify to reviewers as "incremental improvement with architectural foundation for future hardware".

### Path E — Upgrade demo hardware
Run on M3 Pro / M4 at demo time. Untested, expensive, risks day-of failure if the new machine has different config issues.

---

## 7. Reproducibility

```bash
# Setup (one-time)
python3.11 -m venv .venv-smoke
source .venv-smoke/bin/activate
pip install -r requirements-smoke.txt
mkdir -p models/gemma-4-e2b
curl -L -o models/gemma-4-e2b/gemma-4-E2B-it.litertlm \
  "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm"

# Regenerate fixtures (if ElevenLabs voice changes)
python -m scripts._gen_multimodal_fixtures

# Re-run gates
python -m scripts._gate1_collect          # ~60s (Gemma + Gemini responses)
python -m scripts._gate1_gemini_retry     # ~40s (if model name deprecated)
python -m scripts._gate1_rate             # <1s (applies Claude's ratings)
python -m scripts.smoke_gemma_multimodal  # ~80s (5 × multimodal)
python -m scripts.bench_latency           # ~5 min (20 iterations)
```

Ratings used in `_gate1_rate.py` are hard-coded with rationale comments; a human operator should re-rate to validate before acting on the PASS/FAIL decision.

## 8. Test Harness Health

All 19 unit tests in `scripts/tests/` pass in 0.03 s — `pytest scripts/tests/ -v`.

```
scripts/tests/test_smoke_helpers.py ............ 13 passed
scripts/tests/test_korean_prompts_fixture.py ... 6 passed
```

The `_smoke_helpers` module (rating aggregation, percentile calculation, WER, pass/fail formatting) is pure and independent of LiteRT-LM or any network service, ensuring the harness is correct even if the gates themselves fail.
