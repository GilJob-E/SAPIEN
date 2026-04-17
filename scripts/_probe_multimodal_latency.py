"""
Quick probe: measure Gemma 4 E2B multimodal inference latency on M2.

Uses synthetic audio (16kHz mono silence) + synthetic JPEG (512x384) to avoid
needing real recorded fixtures. Latency depends on input SIZE not content,
so synthetic data gives realistic timing numbers.

Purpose: inform Gate #1 decision — if multimodal latency is acceptable,
Gate #1's marginal text-quality issues are fixable via prompting in M2.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).parent.parent
MODEL_PATH = REPO_ROOT / "models" / "gemma-4-e2b" / "gemma-4-E2B-it.litertlm"
TMP = REPO_ROOT / "scripts" / "_results"
TMP.mkdir(exist_ok=True)

AUDIO_PATH = TMP / "_probe_audio.wav"
IMAGE_PATH = TMP / "_probe_image.jpg"


def make_synthetic_audio(path: Path, duration_s: float = 5.0, sample_rate: int = 16000) -> None:
    """Create a WAV file of the given duration (16kHz mono int16)."""
    n_samples = int(duration_s * sample_rate)
    # Low-amplitude sine wave — gives the audio encoder something to chew on
    # rather than pure silence which some codecs may short-circuit.
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    signal = (np.sin(2 * np.pi * 220 * t) * 0.1 * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())


def make_synthetic_image(path: Path, width: int = 512, height: int = 384) -> None:
    """Create a JPEG file the size of a typical webcam frame (quality 70)."""
    # Mostly-mid-grey with a few colored blobs so the JPEG isn't pathologically small.
    arr = np.full((height, width, 3), 128, dtype=np.uint8)
    arr[50:200, 50:200] = [200, 120, 80]
    arr[220:350, 280:480] = [80, 160, 200]
    Image.fromarray(arr, "RGB").save(path, "JPEG", quality=70)


def main() -> int:
    from litert_lm import Engine, Backend  # type: ignore

    print("[1/5] Generating synthetic fixtures...")
    make_synthetic_audio(AUDIO_PATH, duration_s=5.0)
    make_synthetic_image(IMAGE_PATH)
    print(f"      audio: {AUDIO_PATH.stat().st_size // 1024} KB (5s, 16kHz mono)")
    print(f"      image: {IMAGE_PATH.stat().st_size // 1024} KB (512x384 JPEG)")

    print("[2/5] Loading Gemma 4 E2B with audio+vision backends...")
    t0 = time.perf_counter()
    engine = Engine(
        model_path=str(MODEL_PATH),
        backend=Backend.CPU,
        audio_backend=Backend.CPU,
        vision_backend=Backend.CPU,  # GPU support is upcoming per docs
    )
    load_s = time.perf_counter() - t0
    print(f"      loaded in {load_s:.2f}s")

    print("[3/5] Warmup inference (not counted)...")
    t0 = time.perf_counter()
    with engine.create_conversation() as conv:
        warm_msg = {
            "role": "user",
            "content": [
                {"type": "audio", "path": str(AUDIO_PATH)},
                {"type": "image", "path": str(IMAGE_PATH)},
                {"type": "text", "text": "이 음성과 이미지를 간단히 설명해주세요."},
            ],
        }
        _ = conv.send_message(warm_msg)
    warm_s = time.perf_counter() - t0
    print(f"      warmup took {warm_s:.2f}s")

    print("[4/5] Measuring 5 multimodal inferences...")
    latencies = []
    for i in range(1, 6):
        t0 = time.perf_counter()
        with engine.create_conversation() as conv:
            msg = {
                "role": "user",
                "content": [
                    {"type": "audio", "path": str(AUDIO_PATH)},
                    {"type": "image", "path": str(IMAGE_PATH)},
                    {"type": "text", "text": "이 음성과 이미지를 간단히 설명해주세요."},
                ],
            }
            resp = conv.send_message(msg)
        s = time.perf_counter() - t0
        latencies.append(s)
        text = ""
        try:
            text = resp["content"][0]["text"][:60].replace("\n", " ")
        except Exception:
            text = str(resp)[:60]
        print(f"      [{i}/5] {s:.2f}s  preview: {text}")

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95_idx = max(0, int(len(latencies_sorted) * 0.95) - 1)
    p95 = latencies_sorted[p95_idx]
    mean = sum(latencies) / len(latencies)

    out = {
        "probe": "multimodal_latency",
        "model": "gemma-4-E2B-it.litertlm",
        "backend": "CPU",
        "audio_backend": "CPU",
        "vision_backend": "CPU",
        "audio_duration_s": 5.0,
        "image_size_kb": IMAGE_PATH.stat().st_size // 1024,
        "load_s": round(load_s, 2),
        "warmup_s": round(warm_s, 2),
        "steady_state": {
            "mean_s": round(mean, 2),
            "p50_s": round(p50, 2),
            "p95_s": round(p95, 2),
            "raw_s": [round(s, 2) for s in latencies],
        },
    }
    (TMP / "multimodal_probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[5/5] Summary (5s audio + 512x384 JPEG + Korean prompt):")
    print(f"      load     {load_s:.2f}s")
    print(f"      warmup   {warm_s:.2f}s")
    print(f"      mean     {mean:.2f}s")
    print(f"      p50      {p50:.2f}s")
    print(f"      p95      {p95:.2f}s")
    print(f"      raw:     {[f'{s:.2f}' for s in latencies]}")

    # Compare against Gate #3 thresholds (spec §8.1)
    P50_TARGET = 2.5
    P95_TARGET = 4.0
    passed = p50 < P50_TARGET and p95 < P95_TARGET
    print(f"\n      Gate #3 preview (same threshold):"
          f" p50 {'<' if p50 < P50_TARGET else '>='} {P50_TARGET}s,"
          f" p95 {'<' if p95 < P95_TARGET else '>='} {P95_TARGET}s"
          f" → {'✅ likely PASS' if passed else '⚠️ borderline/FAIL'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
