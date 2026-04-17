import os
from google import genai
from google.genai import types

_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
MODEL = "gemini-3.1-flash-lite-preview"

_LANG_NAMES = {
    "ko": "Korean", "en": "English", "ja": "Japanese",
    "zh": "Chinese", "es": "Spanish", "fr": "French",
    "de": "German", "pt": "Portuguese", "ru": "Russian",
    "it": "Italian",
}


class Speech2Text:
    def recognize_from_file(self, filename, language="ko"):
        print("[STT] Gemini 인식 시작...")
        try:
            lang_code = language[:2] if language else "ko"
            lang_name = _LANG_NAMES.get(lang_code, "Korean")

            with open(str(filename), "rb") as f:
                audio_bytes = f.read()

            response = _client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                    f"Transcribe this audio in {lang_name}. Output only the transcribed text, nothing else.",
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=500,
                    temperature=0.1,
                ),
            )
            text = response.text.strip()
            if not text:
                print("[STT] 인식 결과 없음")
                return "..."
            print(f"[STT] 인식: {text}")
            return text
        except Exception as e:
            print(f"[STT] Gemini 인식 실패: {e}")
            return "..."
