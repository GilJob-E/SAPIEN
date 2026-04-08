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
import wave
from .keys import *
from .globals import *
from elevenlabs import ElevenLabs, VoiceSettings


EMOTION_SETTINGS = {
    'NEUTRAL':   VoiceSettings(stability=0.7, similarity_boost=0.75, style=0.0),
    'HAPPY':     VoiceSettings(stability=0.5, similarity_boost=0.75, style=0.5),
    'SAD':       VoiceSettings(stability=0.9, similarity_boost=0.75, style=0.2),
    'ANGRY':     VoiceSettings(stability=0.3, similarity_boost=0.75, style=0.7),
    'SURPRISED': VoiceSettings(stability=0.4, similarity_boost=0.75, style=0.6),
    'AFRAID':    VoiceSettings(stability=0.8, similarity_boost=0.75, style=0.3),
    'DISGUSTED': VoiceSettings(stability=0.8, similarity_boost=0.75, style=0.4),
}

_eleven_client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY", ""))


class Text2Speech:
    def __init__(self):
        self.voice_id = None
        self.audiodir = None
        self.audiofile = None
        self.language = None
        self.pronoun = None

    def set_audio(self, bot_name, pronoun, language='en-US', audiodir=None, audiofile=None):
        self.language = language
        self.pronoun = pronoun
        self.voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "")
        self.audiodir = audiodir
        self.audiofile = audiofile
        print(f"[TTS] ElevenLabs voice_id: {self.voice_id}")

    def create_wav(self, text, emo='NEUTRAL', ssml=True):
        global wav_lock, emotion_ready
        text = text.strip()
        if not text:
            text = languages[self.language]['connection_interruption']

        voice_settings = EMOTION_SETTINGS.get(emo.upper(), EMOTION_SETTINGS['NEUTRAL'])

        with wav_lock:
            emotion_ready[0] = emo

            try:
                audio_generator = _eleven_client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    text=text,
                    model_id="eleven_multilingual_v2",
                    output_format="pcm_16000",
                    voice_settings=voice_settings,
                )
                pcm_data = b"".join(audio_generator)

                # PCM → WAV (16kHz, 16-bit, mono)
                with wave.open(str(self.audiofile), 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(pcm_data)

                print(f"[TTS] WAV 생성 완료: {len(pcm_data)} bytes, emotion={emo}")

            except Exception as e:
                print(f"[TTS] ElevenLabs 호출 실패: {e}")
                # 무음 WAV 생성 (폴백)
                silence = b'\x00\x00' * 16000  # 1초 무음
                with wave.open(str(self.audiofile), 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(silence)
