"""
Step 0 Smoke Test - Gemini / ElevenLabs / Whisper API 검증
실행: python -m start_app.tests.smoke_test_step0
"""

import os
import sys
import time

# keys.py 로드 (환경변수 세팅)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dialogue_manager.keys import *


def test_gemini_vision():
    """Gemini 3 Flash - 모델 검증 + Vision API latency 측정"""
    from google import genai
    from PIL import Image
    import io

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    print("[Gemini] Client 초기화 성공")

    # 더미 이미지로 Vision API latency 측정
    img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img = Image.open(buf)

    start = time.time()
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[img, "이 이미지를 간단히 설명해주세요."],
    )
    elapsed = time.time() - start

    print(f"[Gemini] Vision API 응답: {response.text[:100]}...")
    print(f"[Gemini] Latency: {elapsed:.2f}s {'✅ PASS' if elapsed <= 2.0 else '❌ FAIL (>2s)'}")
    return elapsed <= 2.0


def test_elevenlabs_korean():
    """ElevenLabs - eleven_multilingual_v2 한국어 음성 생성 테스트"""
    from elevenlabs import ElevenLabs

    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    voice_id = os.environ["ELEVENLABS_VOICE_ID"]

    start = time.time()
    audio_generator = client.text_to_speech.convert(
        voice_id=voice_id,
        text="안녕하세요, 면접을 시작하겠습니다.",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    # generator를 bytes로 변환
    audio_bytes = b"".join(audio_generator)
    elapsed = time.time() - start

    print(f"[ElevenLabs] 한국어 TTS 생성 성공: {len(audio_bytes)} bytes")
    print(f"[ElevenLabs] Latency: {elapsed:.2f}s")

    # 결과 파일 저장 (확인용)
    out_path = os.path.join(os.path.dirname(__file__), "elevenlabs_test.mp3")
    with open(out_path, "wb") as f:
        f.write(audio_bytes)
    print(f"[ElevenLabs] 저장: {out_path} ✅ PASS")
    return True


def test_whisper_korean():
    """Whisper base 모델 - 다운로드 + CPU latency 측정"""
    import whisper

    print("[Whisper] base 모델 로드 중...")
    start = time.time()
    model = whisper.load_model("base")
    load_time = time.time() - start
    print(f"[Whisper] 모델 로드 완료: {load_time:.2f}s")

    # ElevenLabs 테스트에서 생성된 파일로 테스트 (없으면 스킵)
    test_audio = os.path.join(os.path.dirname(__file__), "elevenlabs_test.mp3")
    if not os.path.exists(test_audio):
        print("[Whisper] 테스트 오디오 없음 - ElevenLabs 테스트 먼저 실행 필요")
        print("[Whisper] 모델 로드만 확인 ✅ PASS")
        return True

    start = time.time()
    result = model.transcribe(test_audio, language="ko")
    elapsed = time.time() - start

    print(f"[Whisper] 한국어 인식 결과: {result['text']}")
    print(f"[Whisper] CPU Latency: {elapsed:.2f}s ✅ PASS")
    return True


if __name__ == "__main__":
    results = {}

    print("=" * 60)
    print("Step 0 Smoke Test")
    print("=" * 60)

    tests = [
        ("Gemini Vision", test_gemini_vision),
        ("ElevenLabs Korean TTS", test_elevenlabs_korean),
        ("Whisper Korean STT", test_whisper_korean),
    ]

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"❌ FAIL: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\n전체: {'✅ ALL PASS' if all_passed else '❌ SOME FAILED'}")
    sys.exit(0 if all_passed else 1)
