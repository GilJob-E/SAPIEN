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
import json
import time
from google import genai
from google.genai import types


_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
MODEL = "gemini-2.5-flash"


class LLM:
    def __init__(self, instruction=""):
        self.system_messages = []
        if instruction:
            self.system_messages.append(instruction)
        self.max_tokens = 150

    def add_content(self, content):
        self.system_messages.append(content)

    def add_example(self, format):
        self.system_messages.append("Use the following format: " + format)

    def add_system_message(self, message):
        self.system_messages.append(message)

    def get_system_messages(self):
        return "\n".join(self.system_messages) + "\n"

    def ask(self, question, max_tokens=None, temperature=0.7):
        if max_tokens is None:
            max_tokens = self.max_tokens
        system_instruction = self.get_system_messages() if self.system_messages else None

        try:
            response = _client.models.generate_content(
                model=MODEL,
                contents=question,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            return response.text
        except Exception as e:
            print(f"[Gemini] API 호출 실패: {e}")
            # 1회 재시도
            time.sleep(3)
            try:
                response = _client.models.generate_content(
                    model=MODEL,
                    contents=question,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
                return response.text
            except Exception as e2:
                print(f"[Gemini] 재시도 실패: {e2}")
                return "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요."

    def ret_JSON(self, text=""):
        system_instruction = self.get_system_messages() if self.system_messages else None
        turn = 3
        while turn > 0:
            try:
                response = _client.models.generate_content(
                    model=MODEL,
                    contents=text,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        max_output_tokens=self.max_tokens,
                        temperature=0.7,
                        response_mime_type="application/json",
                    ),
                )
                ret = json.loads(response.text)
                if ret:
                    print(f"ret_JSON: {ret}")
                    return ret
            except Exception:
                print("Error parsing generated JSON. Trying again.")
            turn -= 1
        return {}

    def check_generation(self, text):
        check_prompt = (
            "Respond YES or NO. Does this Generated Text follow the Instructions properly?\n"
            f"Instructions:\n----\n{self.get_system_messages()}----\n"
            f"Generated Text:\n----\n{text}\n----\n"
        )
        response_text = self.ask(check_prompt)
        return response_text[:3].lower().strip() == "yes"

    # Backward-compatible aliases
    def chat_generate(self, text):
        return self.ask(text)

    def ask_chat(self, question, max_tokens=150):
        return self.ask(question, max_tokens=max_tokens)

    def temp_ask_chat(self, question):
        return self.ask(question, max_tokens=500)

    def ask_gpt4(self, question, max_tokens=150):
        return self.ask(question, max_tokens=max_tokens)
