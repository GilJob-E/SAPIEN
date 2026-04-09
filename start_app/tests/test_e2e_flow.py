"""E2E 흐름 및 에러 폴백 시나리오 테스트.

API 키가 필요한 테스트는 @pytest.mark.api로 표시.
기본 실행 시 API 없이 동작하는 테스트만 실행.
"""
import os
import subprocess
import pytest


class TestSmoke:
    def test_no_openai_imports(self):
        """코드에서 openai 패키지 import가 없는지 확인."""
        result = subprocess.run(
            ["grep", "-r", "import openai", "start_app/",
             "--include=*.py", "--exclude-dir=tests", "-l"],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), "../.."),
        )
        assert result.stdout.strip() == "", f"openai import 발견: {result.stdout}"

    def test_no_deepface_imports(self):
        """코드에서 deepface 패키지 import가 없는지 확인."""
        result = subprocess.run(
            ["grep", "-rE", "import deepface|from deepface", "start_app/",
             "--include=*.py", "--exclude-dir=tests", "-l"],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), "../.."),
        )
        assert result.stdout.strip() == "", f"deepface import 발견: {result.stdout}"

    def test_no_cv2_imports(self):
        """코드에서 cv2 import가 없는지 확인."""
        result = subprocess.run(
            ["grep", "-r", "import cv2", "start_app/",
             "--include=*.py", "--exclude-dir=tests", "-l"],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), "../.."),
        )
        assert result.stdout.strip() == "", f"cv2 import 발견: {result.stdout}"

    def test_google_genai_importable(self):
        """google-genai 패키지가 설치되어 있는지 확인."""
        import google.genai
        assert hasattr(google.genai, "Client")

    def test_elevenlabs_importable(self):
        """elevenlabs 패키지가 설치되어 있는지 확인."""
        from elevenlabs import ElevenLabs
        assert ElevenLabs is not None

    def test_no_whisper_dependency(self):
        """whisper 의존성이 제거되었는지 확인 (Gemini STT로 교체됨)."""
        import subprocess
        result = subprocess.run(
            ["grep", "-r", "import whisper", "start_app/",
             "--include=*.py", "--exclude-dir=tests", "-l"],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), "../.."),
        )
        assert result.stdout.strip() == "", f"whisper import 발견: {result.stdout}"


class TestErrorFallback:
    def test_camera_denied_no_frame(self):
        """카메라 권한 거부 시: frame_data=None → 감정 분석 스킵."""
        from helper.user_emotion import get_emotion, _DEFAULT_EMOTION
        result = get_emotion(None)
        assert result == _DEFAULT_EMOTION

    def test_stt_api_failure_returns_ellipsis(self):
        """STT API 실패 → '...' 반환, 크래시 없음."""
        from unittest.mock import patch, MagicMock
        # meeting 테스트가 모듈을 mock으로 교체하므로 직접 재로드
        import importlib, sys
        if "dialogue_manager.speech2text" in sys.modules:
            del sys.modules["dialogue_manager.speech2text"]
        import dialogue_manager.speech2text as stt_mod
        importlib.reload(stt_mod)

        with patch.object(stt_mod, "_client") as mock_client:
            mock_client.models.generate_content.side_effect = Exception("API error")
            stt = stt_mod.Speech2Text()
            result = stt.recognize_from_file("nonexistent.wav", "ko-KR")
            assert result == "..."

    def test_gemini_api_timeout_no_crash(self):
        """Gemini API 타임아웃 → 기본값 반환, 크래시 없음."""
        from unittest.mock import patch, MagicMock
        with patch("helper.user_emotion._client") as mock_client:
            mock_client.models.generate_content.side_effect = TimeoutError("timeout")
            from helper.user_emotion import get_emotion, _DEFAULT_EMOTION
            result = get_emotion("dGVzdA==")  # "test" in base64
            assert result == _DEFAULT_EMOTION


@pytest.mark.api
class TestAPISmoke:
    """API 키가 필요한 통합 테스트. pytest -m api 로 실행."""

    def test_gemini_api_key_valid(self):
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        assert api_key, "GOOGLE_API_KEY 환경변수가 설정되지 않음"

    def test_elevenlabs_api_key_valid(self):
        api_key = os.environ.get("ELEVENLABS_API_KEY", "")
        assert api_key, "ELEVENLABS_API_KEY 환경변수가 설정되지 않음"
