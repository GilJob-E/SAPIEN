# Code authors: Masum Hasan, Cengiz Ozel, Sammy Potter
# ROC-HCI Lab, University of Rochester
# Copyright (c) 2023 University of Rochester

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import os
import whisper
import torch
import warnings

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model_size = "base" if _device == "cuda" else "base"
print(f"[STT] Whisper 모델 로드: {_model_size} (device={_device})")
_whisper_model = whisper.load_model(_model_size, device=_device)
print("[STT] Whisper 모델 로드 완료")


class Speech2Text:
    def recognize_from_file(self, filename, language="ko"):
        print("[STT] Whisper 인식 시작...")
        try:
            # language 파라미터: "en-US" → "en", "ko-KR" → "ko"
            lang_code = language[:2] if language else "ko"
            result = _whisper_model.transcribe(
                str(filename),
                language=lang_code,
                fp16=(_device == "cuda"),
            )
            text = result["text"].strip()
            if not text:
                print("[STT] 인식 결과 없음")
                return "..."
            print(f"[STT] 인식: {text}")
            return text
        except Exception as e:
            print(f"[STT] Whisper 인식 실패: {e}")
            return "..."
