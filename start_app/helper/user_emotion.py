from google import genai
from google.genai import types
import os, json, base64

_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
MODEL = "gemini-3.1-flash-lite-preview"

_DEFAULT_EMOTION = {"confidence": "medium", "engagement": "medium", "note": ""}
_VALID_LEVELS = {"high", "medium", "low"}
_MAX_FRAME_BASE64_LEN = 1_000_000  # ~750KB decoded JPEG

def _sanitize_emotion(result):
    """Gemini 응답을 허용된 값으로 제한."""
    confidence = result.get("confidence", "medium")
    engagement = result.get("engagement", "medium")
    note = result.get("note", "")
    return {
        "confidence": confidence if confidence in _VALID_LEVELS else "medium",
        "engagement": engagement if engagement in _VALID_LEVELS else "medium",
        "note": str(note)[:80].replace("\n", " "),
    }

def get_emotion(frame_base64):
    """웹캠 프레임(base64 JPEG)에서 감정 분석. 실패 시 기본값 반환."""
    if not frame_base64 or len(frame_base64) > _MAX_FRAME_BASE64_LEN:
        return dict(_DEFAULT_EMOTION)
    try:
        response = _client.models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_bytes(
                    data=base64.b64decode(frame_base64),
                    mime_type="image/jpeg",
                ),
                "이 사람의 표정을 분석하세요. JSON으로만 응답: "
                '{"confidence": "high|medium|low", '
                '"engagement": "high|medium|low", '
                '"note": "간단한 관찰"}',
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=200,
                temperature=0.3,
            ),
        )
        result = json.loads(response.text)
        if not isinstance(result, dict):
            return dict(_DEFAULT_EMOTION)
        return _sanitize_emotion(result)
    except Exception as e:
        print(f"[Emotion] Gemini Vision 실패: {e}")
        return dict(_DEFAULT_EMOTION)
