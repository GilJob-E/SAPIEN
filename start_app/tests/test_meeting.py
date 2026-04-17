"""meeting.py 단위 테스트 — Meeting.respond() 및 유틸리티.

globals.py가 모듈 레벨에서 JSON 파일을 읽으므로,
sys.modules에 mock 주입 후 import.
"""
import sys
import os
import time
import json
import pytest
from unittest.mock import MagicMock, patch
from threading import Lock
from pathlib import Path


# ── 무거운 모듈들을 전부 mock 처리 (모듈 로드 전) ─────────────────
def _setup_mocks():
    """meeting.py import 전에 모든 무거운 의존성을 mock."""
    # elevenlabs
    mock_elevenlabs = MagicMock()
    sys.modules["elevenlabs"] = mock_elevenlabs

    # scipy, pylivelinkface, pyglet
    for mod in ["faiss", "scipy", "scipy.io", "scipy.io.wavfile",
                "pylivelinkface", "pylivelinkface.PyLiveLinkFace",
                "pylivelinkface.FaceBlendShape", "pyglet"]:
        sys.modules[mod] = MagicMock()

    # emoji — separate_emotion에서 사용
    mock_emoji = MagicMock()
    mock_emoji.replace_emoji = lambda text, replace='': text
    sys.modules["emoji"] = mock_emoji

    # nltk
    mock_nltk = MagicMock()
    mock_nltk.data.find.return_value = True
    sys.modules["nltk"] = mock_nltk
    mock_tokenize = MagicMock()
    mock_tokenize.sent_tokenize = lambda x: [x]
    sys.modules["nltk.tokenize"] = mock_tokenize

    # globals — 실제 모듈처럼 동작하는 mock 모듈 생성
    import types as _types
    mock_globals = _types.ModuleType("dialogue_manager.globals")
    mock_globals.wav_lock = Lock()
    mock_globals.emotion_ready = ["NEUTRAL"]
    mock_globals.audio_ready_to_send = [False]
    mock_globals.root_path = Path(__file__).parent.parent.absolute()
    mock_globals.local = True
    mock_globals.prerendered = True
    mock_globals.play_audio = False
    mock_globals.send_blendshapes = True
    mock_globals.Path = Path  # meeting.py에서 star import로 사용
    mock_globals.os = os
    mock_globals.json = json
    mock_globals.Lock = Lock
    mock_globals.languages = {
        "ko-KR": {"name": "Korean", "char/token": 3.5, "timeout": 15,
                   "male": [], "female": [],
                   "connection_interruption": "연결이 불안정합니다."},
        "en-US": {"name": "English", "char/token": 4.0, "timeout": 10,
                  "male": [], "female": [],
                  "connection_interruption": "Connection interrupted."},
    }
    mock_globals.emoji_dict = {
        '😊': 'HAPPY_low', '😢': 'SAD_medium', '😡': 'ANGRY_medium',
        '😮': 'SURPRISED_high', '🤢': 'DISGUSTED_high', '😨': 'AFRAID_high',
        '😐': 'NEUTRAL_low', '🙂': 'HAPPY_low',
    }
    mock_globals.Expressions = {
        'NEUTRAL': 0, 'HAPPY': 1, 'SAD': 2, 'ANGRY': 3,
        'SURPRISED': 4, 'DISGUSTED': 5, 'AFRAID': 6,
    }
    sys.modules["dialogue_manager.globals"] = mock_globals

    # keys
    mock_keys = MagicMock()
    sys.modules["dialogue_manager.keys"] = mock_keys

    # speech2text — Meeting.__init__에서 Speech2Text() 호출
    mock_stt = MagicMock()
    mock_stt.Speech2Text = MagicMock
    sys.modules["dialogue_manager.speech2text"] = mock_stt

    # text2speech — Meeting.__init__에서 Text2Speech() 호출
    mock_tts = MagicMock()
    mock_tts.Text2Speech = MagicMock
    sys.modules["dialogue_manager.text2speech"] = mock_tts

    # send_audio_expressive
    sys.modules["dialogue_manager.send_audio_expressive"] = MagicMock()


_setup_mocks()

# 이제 안전하게 import
from dialogue_manager.meeting import Meeting, User, Bot, gemini_chat_call
from dialogue_manager.llm import LLM


class ConcreteMeeting(Meeting):
    """테스트용 concrete Meeting 서브클래스."""
    def ready_prompt(self):
        self.prompt = "Test prompt\n"


@pytest.fixture
def meeting():
    user = User("TestUser", "")
    bot = Bot("TestBot", "", "he")
    m = ConcreteMeeting(user, bot, language="ko-KR")
    m.ready_prompt()
    return m


class TestMeetingRespond:
    @patch("dialogue_manager.meeting.gemini_chat_call")
    def test_respond_without_emotion(self, mock_call, meeting):
        mock_call.return_value = "좋은 답변이네요."
        response, emotion = meeting.respond("안녕하세요", is_emo=True)
        assert "좋은 답변이네요" in response

    @patch("dialogue_manager.meeting.gemini_chat_call")
    def test_respond_with_emotion(self, mock_call, meeting):
        mock_call.return_value = "좋습니다."
        user_emotion = {"confidence": "high", "engagement": "medium", "note": "smiling"}
        response, emotion = meeting.respond("답변입니다", user_emotion=user_emotion)
        assert "confidence=high" in meeting.prompt

    @patch("dialogue_manager.meeting.gemini_chat_call")
    def test_respond_with_none_emotion(self, mock_call, meeting):
        mock_call.return_value = "네."
        meeting.respond("테스트", user_emotion=None)
        assert "[User state:" not in meeting.prompt

    @patch("dialogue_manager.meeting.gemini_chat_call")
    def test_respond_with_non_dict_emotion(self, mock_call, meeting):
        mock_call.return_value = "네."
        meeting.respond("테스트", user_emotion="invalid")
        assert "[User state:" not in meeting.prompt

    @patch("dialogue_manager.meeting.gemini_chat_call")
    def test_respond_with_empty_dict(self, mock_call, meeting):
        mock_call.return_value = "네."
        meeting.respond("테스트", user_emotion={})
        # 빈 dict는 falsy → 감정 주입 안 됨
        assert "[User state:" not in meeting.prompt


class TestTimeWarning:
    @patch("dialogue_manager.meeting.gemini_chat_call")
    def test_time_warning_triggers(self, mock_call, meeting):
        mock_call.return_value = "네."
        meeting.start_time = time.time() - (meeting.max_time_minutes * 60)
        meeting.respond("테스트")
        assert meeting.time_warning_done is True


class TestSeparateEmotion:
    def test_extract_happy_emoji(self, meeting):
        response, emotion = meeting.separate_emotion("좋은 답변 😊")
        assert emotion == "HAPPY"

    def test_no_emoji_returns_neutral(self, meeting):
        response, emotion = meeting.separate_emotion("그냥 텍스트")
        assert emotion == "NEUTRAL"


class TestHistory:
    @patch("dialogue_manager.meeting.gemini_chat_call")
    def test_history_accumulates(self, mock_call, meeting):
        mock_call.return_value = "응답."
        meeting.respond("첫번째")
        meeting.respond("두번째")
        assert len(meeting.history) == 4  # 2 user + 2 bot
