"""user_emotion.py 단위 테스트 — Gemini Vision 감정 분석."""
import json
import pytest
from unittest.mock import patch, MagicMock
from helper.user_emotion import (
    get_emotion,
    _sanitize_emotion,
    _DEFAULT_EMOTION,
    _VALID_LEVELS,
    _MAX_FRAME_BASE64_LEN,
)


class TestSanitizeEmotion:
    def test_valid_values_pass_through(self):
        result = _sanitize_emotion({"confidence": "high", "engagement": "low", "note": "OK"})
        assert result == {"confidence": "high", "engagement": "low", "note": "OK"}

    def test_invalid_confidence_falls_back(self):
        result = _sanitize_emotion({"confidence": "very_high", "engagement": "low", "note": ""})
        assert result["confidence"] == "medium"

    def test_invalid_engagement_falls_back(self):
        result = _sanitize_emotion({"confidence": "high", "engagement": "강함", "note": ""})
        assert result["engagement"] == "medium"

    def test_note_truncated_at_80_chars(self):
        long_note = "a" * 200
        result = _sanitize_emotion({"confidence": "high", "engagement": "high", "note": long_note})
        assert len(result["note"]) == 80

    def test_note_newlines_stripped(self):
        result = _sanitize_emotion({"confidence": "high", "engagement": "high", "note": "line1\nline2"})
        assert "\n" not in result["note"]

    def test_missing_keys_get_defaults(self):
        result = _sanitize_emotion({})
        assert result["confidence"] == "medium"
        assert result["engagement"] == "medium"
        assert result["note"] == ""

    def test_note_non_string_converted(self):
        result = _sanitize_emotion({"confidence": "high", "engagement": "high", "note": 12345})
        assert result["note"] == "12345"


class TestGetEmotion:
    @patch("helper.user_emotion._client")
    def test_valid_response(self, mock_client, sample_base64_jpeg):
        mock_resp = MagicMock()
        mock_resp.text = '{"confidence": "high", "engagement": "low", "note": "focused"}'
        mock_client.models.generate_content.return_value = mock_resp

        result = get_emotion(sample_base64_jpeg)
        assert result["confidence"] == "high"
        assert result["engagement"] == "low"
        assert mock_client.models.generate_content.called

    def test_none_input_returns_default(self):
        result = get_emotion(None)
        assert result == _DEFAULT_EMOTION

    def test_empty_string_returns_default(self):
        result = get_emotion("")
        assert result == _DEFAULT_EMOTION

    def test_oversized_frame_returns_default(self):
        big_frame = "A" * (_MAX_FRAME_BASE64_LEN + 1)
        result = get_emotion(big_frame)
        assert result == _DEFAULT_EMOTION

    @patch("helper.user_emotion._client")
    def test_api_failure_returns_default(self, mock_client, sample_base64_jpeg):
        mock_client.models.generate_content.side_effect = Exception("API down")
        result = get_emotion(sample_base64_jpeg)
        assert result == _DEFAULT_EMOTION

    @patch("helper.user_emotion._client")
    def test_non_dict_response_returns_default(self, mock_client, sample_base64_jpeg):
        mock_resp = MagicMock()
        mock_resp.text = '["not", "a", "dict"]'
        mock_client.models.generate_content.return_value = mock_resp

        result = get_emotion(sample_base64_jpeg)
        assert result == _DEFAULT_EMOTION

    @patch("helper.user_emotion._client")
    def test_invalid_json_returns_default(self, mock_client, sample_base64_jpeg):
        mock_resp = MagicMock()
        mock_resp.text = "not json at all"
        mock_client.models.generate_content.return_value = mock_resp

        result = get_emotion(sample_base64_jpeg)
        assert result == _DEFAULT_EMOTION

    @patch("helper.user_emotion._client")
    def test_sanitization_applied(self, mock_client, sample_base64_jpeg):
        mock_resp = MagicMock()
        mock_resp.text = '{"confidence": "INVALID", "engagement": "high", "note": "ok"}'
        mock_client.models.generate_content.return_value = mock_resp

        result = get_emotion(sample_base64_jpeg)
        assert result["confidence"] == "medium"  # sanitized
        assert result["engagement"] == "high"  # valid, kept

    def test_default_emotion_is_independent_copy(self):
        """각 호출이 독립적인 dict를 반환하는지 확인 (글로벌 캐시 누출 방지)."""
        r1 = get_emotion(None)
        r2 = get_emotion(None)
        r1["confidence"] = "modified"
        assert r2["confidence"] == "medium"  # r2는 영향 없음
