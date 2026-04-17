"""
_gen_multimodal_fixtures.py — Generate synthetic Korean audio + photo for Gate #2.

Uses ElevenLabs (multilingual_v2) to synthesise Korean speech from known
transcripts. Photo is a small solid-colour JPEG matching production frame
size. This replaces the "operator records 5 audio samples" manual step in
the original plan (Task 5) — audio is fully determined by the transcripts
here, so reproducible across machines.

Produces:
  scripts/fixtures/audio_samples/{self_intro,motivation,experience,weakness,question}.wav
  scripts/fixtures/test_photo.jpg

Labels (expected_emotions.json) are written by hand elsewhere — this script
doesn't generate ground truth.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from elevenlabs.client import ElevenLabs

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
import start_app.dialogue_manager.keys  # noqa: F401 — sets env vars

from PIL import Image
import numpy as np

FIXTURES = REPO_ROOT / "scripts" / "fixtures"
AUDIO_DIR = FIXTURES / "audio_samples"
PHOTO = FIXTURES / "test_photo.jpg"

# Transcripts exactly as they appear in expected_emotions.json — this is the
# ground truth the Gate #2 WER is computed against.
TRANSCRIPTS = [
    ("self_intro",  "안녕하세요, 저는 김진우라고 합니다. 컴퓨터공학을 전공하고 있습니다."),
    ("motivation",  "이 포지션에 지원한 이유는 실시간 AI 시스템에 관심이 많기 때문입니다."),
    ("experience",  "최근에 저는 멀티모달 면접 플랫폼을 개발하고 있습니다."),
    ("weakness",    "제 단점은 한 가지 일에 너무 몰입해서 다른 일을 놓치는 경우가 있다는 점입니다."),
    ("question",    "혹시 팀의 엔지니어링 문화에 대해 조금 더 알려주실 수 있을까요?"),
]


def _tts_to_wav(client: ElevenLabs, voice_id: str, text: str, out_path: Path) -> None:
    """Stream ElevenLabs TTS in 16 kHz PCM and wrap in a WAV container (no ffmpeg)."""
    import wave

    pcm_bytes = b""
    stream = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="pcm_16000",  # raw int16 @ 16 kHz mono — no transcode needed
    )
    for chunk in stream:
        pcm_bytes += chunk

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(16000)
        wf.writeframes(pcm_bytes)


def make_photo(path: Path, width: int = 512, height: int = 384) -> None:
    """Synthetic neutral-tone image matching production frame geometry."""
    arr = np.full((height, width, 3), 180, dtype=np.uint8)  # light grey
    # Add a face-like oval of warm tone in the centre so vision encoder has
    # something to latch onto (without claiming it's a real face for Gate #2
    # emotion scoring; emotion labels are neutral/medium anyway).
    cy, cx = height // 2, width // 2
    yy, xx = np.ogrid[:height, :width]
    mask = ((yy - cy) / 90) ** 2 + ((xx - cx) / 70) ** 2 <= 1
    arr[mask] = [220, 200, 175]
    Image.fromarray(arr, "RGB").save(path, "JPEG", quality=80)


def main() -> int:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURES.mkdir(parents=True, exist_ok=True)

    print("[1/3] Preparing ElevenLabs client...")
    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "Xb7hH8MSUJpSbSDYk0k2")
    print(f"      voice_id = {voice_id}")

    print(f"[2/3] Generating {len(TRANSCRIPTS)} Korean audio samples...")
    for name, text in TRANSCRIPTS:
        out_path = AUDIO_DIR / f"{name}.wav"
        print(f"      [{name}] → {out_path.name}")
        _tts_to_wav(client, voice_id, text, out_path)
        print(f"                 {out_path.stat().st_size // 1024} KB")

    print("[3/3] Generating test photo...")
    make_photo(PHOTO)
    print(f"      {PHOTO.name}: {PHOTO.stat().st_size // 1024} KB")

    print("\n[OK] fixtures ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
