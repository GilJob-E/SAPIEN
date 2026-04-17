"""
smoke_gemma_multimodal.py — Gate #2 from design spec §8.1.

For each of 5 audio samples + 1 photo:
  - Invoke Gemma 4 E2B with audio + image
  - Ask it to transcribe + produce emotion JSON (production schema)
  - Compare transcript WER to expected_transcript
  - Compare emotion confidence+engagement to expected labels

PASS criteria (spec §8.1 Gate #2):
  - mean WER across 5 samples <= 0.10 (>=90% accuracy)
  - emotion (confidence + engagement) exact match on ≥ 3 of 5 samples

LiteRT-LM API notes:
- Engine accepts audio_backend / vision_backend keywords (currently CPU only;
  GPU support is upcoming per Google AI Edge docs).
- send_message accepts a {"role", "content"} dict where content is a list of
  parts with {"type": "audio"|"image"|"text", "path"|"text": ...}.
- Response: {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
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

# Pass/fail thresholds (spec §8.1 Gate #2)
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


def _extract_text(response: dict) -> str:
    """Unwrap LiteRT-LM response dict to plain text."""
    try:
        return response["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return str(response)


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

    from litert_lm import Engine, Backend  # type: ignore

    console.print("Loading model with audio + vision backends...")
    engine = Engine(
        model_path=str(MODEL_PATH),
        backend=Backend.CPU,
        audio_backend=Backend.CPU,
        vision_backend=Backend.CPU,
    )
    console.print("[green]Model ready.[/green]\n")

    labels = json.loads(LABELS.read_text(encoding="utf-8"))["samples"]

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
        with engine.create_conversation() as conv:
            msg = {
                "role": "user",
                "content": [
                    {"type": "audio", "path": str(audio_path)},
                    {"type": "image", "path": str(PHOTO)},
                    {"type": "text", "text": PROMPT},
                ],
            }
            raw_resp = conv.send_message(msg)
        raw = _extract_text(raw_resp)
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
