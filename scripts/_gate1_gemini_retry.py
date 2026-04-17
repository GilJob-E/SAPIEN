"""Re-run Gemini baseline on the raw-responses JSON using gemini-2.5-flash."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
import start_app.dialogue_manager.keys  # noqa: F401 — sets env vars

from google import genai  # type: ignore
from google.genai import types  # type: ignore

RAW = Path(__file__).parent / "_results" / "gate1_responses_raw.json"
MODEL = "gemini-2.5-flash"

SYSTEM = (
    "당신은 한국어 면접관입니다. "
    "정중한 존댓말로, 2-3문장 이내로 간결하게 응답하세요. "
    "면접 주제에서 벗어나지 마세요."
)


def main() -> int:
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    data = json.loads(RAW.read_text(encoding="utf-8"))
    print(f"Loaded {len(data['responses'])} entries, retrying Gemini with {MODEL}")

    for i, entry in enumerate(data["responses"], start=1):
        t0 = time.perf_counter()
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=entry["prompt_text"],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    max_output_tokens=1024,  # gemini-2.5-flash needs room for thinking tokens + output
                    temperature=0.7,
                ),
            )
            text = (resp.text or "").strip()
            dur = time.perf_counter() - t0
            print(f"  [{i:2d}/{len(data['responses'])}] {entry['id']} ({dur:.1f}s): {text[:60]}")
            entry["gemini_response"] = text
            entry["gemini_latency_s"] = round(dur, 2)
            entry["gemini_model"] = MODEL
        except Exception as e:
            print(f"  [{i:2d}/{len(data['responses'])}] {entry['id']} FAILED: {e}")
            return 1

    data["gemini_model"] = MODEL
    data["gemini_retried_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    RAW.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Updated {RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
