"""
_gate1_collect.py — Gate #1 response collection (non-interactive).

Runs Gemma 4 E2B + Gemini 2.0 Flash on each of 10 Korean prompts and dumps
all responses to JSON. Rating is done separately from the collected responses.

Temporary helper for Claude-controller to score Gate #1 without a human at
the keyboard. Will be deleted after Gate #1 completes.
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Load API keys as environment variables.
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
import start_app.dialogue_manager.keys  # noqa: F401 — side effect: sets env vars

from scripts.smoke_gemma_korean import (  # reuse lazy loaders
    FIXTURE,
    MODEL_PATH,
    RESULTS_DIR,
    _load_gemma,
)


def _load_gemini():
    """Minimal direct Gemini wrapper — avoids meeting.py's heavy transitive imports.

    Replicates gemini_chat_call() from start_app/dialogue_manager/meeting.py
    so we can use the production baseline without installing pyglet, pandas,
    elevenlabs, etc. in the smoke venv.
    """
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
    import os

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    MODEL = "gemini-2.0-flash"

    def gen(user_msg: str) -> str:
        system_instruction = (
            "당신은 한국어 면접관입니다. "
            "정중한 존댓말로, 2-3문장 이내로 간결하게 응답하세요. "
            "면접 주제에서 벗어나지 마세요."
        )
        resp = client.models.generate_content(
            model=MODEL,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=200,
                temperature=0.7,
            ),
        )
        return (resp.text or "").strip()

    return gen

OUT_PATH = Path(__file__).parent / "_results" / "gate1_responses_raw.json"


def main() -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"[1/4] Loading prompts from {FIXTURE.name}...")
    prompts = json.loads(FIXTURE.read_text(encoding="utf-8"))["prompts"]
    print(f"      {len(prompts)} prompts loaded")

    print(f"[2/4] Loading Gemma 4 E2B from {MODEL_PATH.name}...")
    t0 = time.perf_counter()
    gemma = _load_gemma()
    print(f"      Gemma ready in {time.perf_counter()-t0:.1f}s")

    print(f"[3/4] Loading Gemini 2.0 Flash baseline...")
    t0 = time.perf_counter()
    gemini = _load_gemini()
    print(f"      Gemini ready in {time.perf_counter()-t0:.1f}s")

    print(f"[4/4] Running {len(prompts)} prompts × 2 models...")
    random.seed(42)
    collected = []
    for i, p in enumerate(prompts, start=1):
        print(f"\n  [{i:2d}/{len(prompts)}] {p['id']} ({p['category']})")
        print(f"       Q: {p['text'][:60]}")

        t0 = time.perf_counter()
        try:
            gemma_resp = gemma(p["text"])
            gemma_s = time.perf_counter() - t0
            print(f"       Gemma  ({gemma_s:.1f}s): {gemma_resp[:80].replace(chr(10), ' / ')}")
        except Exception as e:
            gemma_resp = f"[ERROR: {e}]"
            gemma_s = time.perf_counter() - t0
            print(f"       Gemma  FAILED ({gemma_s:.1f}s): {e}")

        t0 = time.perf_counter()
        try:
            gemini_resp = gemini(p["text"])
            gemini_s = time.perf_counter() - t0
            print(f"       Gemini ({gemini_s:.1f}s): {gemini_resp[:80].replace(chr(10), ' / ')}")
        except Exception as e:
            gemini_resp = f"[ERROR: {e}]"
            gemini_s = time.perf_counter() - t0
            print(f"       Gemini FAILED ({gemini_s:.1f}s): {e}")

        # Blind assignment (label which appears as A vs B).
        show_gemma_first = random.random() < 0.5
        label_a = "gemma" if show_gemma_first else "gemini"

        collected.append({
            "id": p["id"],
            "category": p["category"],
            "prompt_text": p["text"],
            "gemma_response": gemma_resp,
            "gemma_latency_s": round(gemma_s, 2),
            "gemini_response": gemini_resp,
            "gemini_latency_s": round(gemini_s, 2),
            "blind_label_a": label_a,
        })

    out = {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "prompts_count": len(prompts),
        "responses": collected,
    }
    # Write BEFORE printing done, in case of SIGSEGV on teardown.
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] Saved {len(collected)} response pairs to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
