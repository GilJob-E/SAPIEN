"""
bench_latency.py — Gate #3 from design spec §8.1.

Measures end-to-end latency on the host M2 MacBook:
  1. Gemma 4 E2B streaming inference (audio + frame → first sentence)
  2. Simulated sentence-boundary detection (emits to TTS on "." "?" "!" or len>40 tokens)
  3. ElevenLabs streaming TTS first-byte (real API call)
  4. Total = Gemma-to-first-sentence + ElevenLabs-first-byte

Runs 20 iterations with the first (warmup) excluded. Reports p50/p95/p99.

PASS criteria (spec §8.1 Gate #3):
  - p50 < 2.5s
  - p95 < 4.0s

LiteRT-LM API notes:
- Conversation.send_message_async(msg) → iterator yielding partial responses
  as they stream in. Each yielded item has {"content": [{"type":"text","text":"..."}]}.
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

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
import start_app.dialogue_manager.keys  # noqa: F401 — sets env vars

from scripts._smoke_helpers import compute_percentiles, format_pass_fail

MODEL_PATH = REPO_ROOT / "models" / "gemma-4-e2b" / "gemma-4-E2B-it.litertlm"
AUDIO = REPO_ROOT / "scripts" / "fixtures" / "audio_samples" / "self_intro.wav"
PHOTO = REPO_ROOT / "scripts" / "fixtures" / "test_photo.jpg"
RESULTS_DIR = REPO_ROOT / "scripts" / "_results"

ITERATIONS = 20
PASS_P50_S = 2.5
PASS_P95_S = 4.0

SENTENCE_TERMINATORS = {".", "!", "?", "。", "?", "!", "\n"}
MAX_BUFFER_CHARS = 120  # if no terminator within this, emit anyway


def _chunk_text(response_dict: dict) -> str:
    """Pull text out of a LiteRT-LM streaming chunk. Safe on partial shapes."""
    try:
        return response_dict["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""


def _gemma_first_sentence(engine, audio_path: Path, image_path: Path, history: list[dict]) -> tuple[str, float, float]:
    """Stream Gemma tokens until a sentence ends.

    Returns (sentence_text, total_seconds, ttft_seconds).
    ttft = time to first (non-empty) chunk.
    """
    t0 = time.perf_counter()
    ttft = None
    seen = ""
    prompt = (
        "이전 대화:\n"
        + "\n".join(f"[{t['role']}] {t['text']}" for t in history)
        + "\n\n사용자의 오디오와 이미지를 듣고 보고, 한국어로 자연스럽게 2-3문장으로 답변하세요."
    )
    msg = {
        "role": "user",
        "content": [
            {"type": "audio", "path": str(audio_path)},
            {"type": "image", "path": str(image_path)},
            {"type": "text", "text": prompt},
        ],
    }
    with engine.create_conversation() as conv:
        for chunk in conv.send_message_async(msg):
            text = _chunk_text(chunk)
            if not text:
                continue
            if ttft is None and text.strip():
                ttft = time.perf_counter() - t0
            # Chunks may be cumulative OR delta depending on backend; detect which.
            if text.startswith(seen):
                new = text[len(seen):]
                seen = text
            else:
                new = text
                seen = seen + text
            # Emit sentence boundary
            last_char = seen[-1] if seen else ""
            if last_char in SENTENCE_TERMINATORS or len(seen) > MAX_BUFFER_CHARS:
                return seen, time.perf_counter() - t0, (ttft or 0.0)
    return seen, time.perf_counter() - t0, (ttft or 0.0)


def _elevenlabs_first_byte(text: str) -> float:
    """Call ElevenLabs streaming TTS, return seconds until first audio byte."""
    from elevenlabs.client import ElevenLabs  # type: ignore

    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "Xb7hH8MSUJpSbSDYk0k2")
    t0 = time.perf_counter()
    stream = client.text_to_speech.convert(
        text=text[:200] if text else "안녕하세요.",
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    for _ in stream:
        return time.perf_counter() - t0
    return time.perf_counter() - t0


def main() -> int:
    console = Console()
    RESULTS_DIR.mkdir(exist_ok=True)
    console.rule("[bold]Gate #3 — M2 MacBook latency benchmark[/bold]")

    console.print("Loading Gemma with multimodal backends...")
    from litert_lm import Engine, Backend  # type: ignore

    engine = Engine(
        model_path=str(MODEL_PATH),
        backend=Backend.CPU,
        audio_backend=Backend.CPU,
        vision_backend=Backend.CPU,
    )

    console.print("Warmup iteration (not counted)...")
    _gemma_first_sentence(engine, AUDIO, PHOTO, history=[])

    history = [
        {"role": "assistant", "text": "안녕하세요, 자기소개 부탁드립니다."},
        {"role": "user", "text": "안녕하세요. 저는 김진우입니다."},
        {"role": "assistant", "text": "네, 반갑습니다. 지원 동기를 말씀해 주세요."},
    ]

    totals: list[float] = []
    gemmas: list[float] = []
    ttfts: list[float] = []
    ttses: list[float] = []

    for i in range(1, ITERATIONS + 1):
        sentence, gemma_s, ttft = _gemma_first_sentence(engine, AUDIO, PHOTO, history)
        tts_s = _elevenlabs_first_byte(sentence)
        total = gemma_s + tts_s
        gemmas.append(gemma_s)
        ttfts.append(ttft)
        ttses.append(tts_s)
        totals.append(total)
        console.print(
            f"  [{i:2d}/{ITERATIONS}] gemma={gemma_s:.2f}s (ttft={ttft:.2f}s) "
            f"tts={tts_s:.2f}s  total={total:.2f}s"
        )

    g = compute_percentiles(gemmas)
    tt = compute_percentiles(ttfts)
    t = compute_percentiles(ttses)
    tot = compute_percentiles(totals)

    passed = tot["p50"] < PASS_P50_S and tot["p95"] < PASS_P95_S

    table = Table(title="Latency summary (seconds)")
    table.add_column("Stage")
    table.add_column("p50")
    table.add_column("p95")
    table.add_column("p99")
    table.add_row("Gemma TTFT (first token)", f"{tt['p50']:.2f}", f"{tt['p95']:.2f}", f"{tt['p99']:.2f}")
    table.add_row("Gemma first-sentence", f"{g['p50']:.2f}", f"{g['p95']:.2f}", f"{g['p99']:.2f}")
    table.add_row("ElevenLabs first-byte", f"{t['p50']:.2f}", f"{t['p95']:.2f}", f"{t['p99']:.2f}")
    table.add_row("[bold]TOTAL[/bold]", f"{tot['p50']:.2f}", f"{tot['p95']:.2f}", f"{tot['p99']:.2f}")
    console.print(table)

    result = {
        "gate": "3_latency",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "iterations": ITERATIONS,
        "thresholds": {"p50_max_s": PASS_P50_S, "p95_max_s": PASS_P95_S},
        "gemma_first_sentence": g,
        "gemma_ttft": tt,
        "elevenlabs_first_byte": t,
        "total": tot,
        "passed": passed,
        "raw_totals_s": totals,
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
