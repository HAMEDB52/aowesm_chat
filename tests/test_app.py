"""Runs the real app script with the OpenAI model replaced by a fake streaming model."""

from pathlib import Path

import langchain_openai
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "apps.py")
calls = []


class FakeChatOpenAI(GenericFakeChatModel):
    def __init__(self, **kwargs):
        calls.append(kwargs)
        super().__init__(messages=iter([AIMessage(content="Hello from the fake model")]))


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    calls.clear()
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)


def test_missing_key_shows_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = AppTest.from_file(APP).run()
    assert "OPENAI_API_KEY" in at.error[0].value
    assert not calls


def test_chat_round_trip(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("Hi").run()
    assert not at.exception
    assert [m["role"] for m in at.session_state.messages] == ["user", "assistant"]
    assert at.session_state.messages[1]["content"] == "Hello from the fake model"
    assert calls[0]["model"] == "gpt-4o-mini" and calls[0]["api_key"] == "test-key"


def test_clear_chat(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("Hi").run()
    at.sidebar.button[0].click().run()
    assert at.session_state.messages == []
