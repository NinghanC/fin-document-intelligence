from __future__ import annotations

import base64

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from utils import model_clients
from utils.model_clients import BedrockChatModel, DemoChatModel, ResilientChatModel, create_chat_model, has_provider_key


def test_bedrock_provider_requires_explicit_model_id(monkeypatch):
    monkeypatch.setattr(model_clients.settings, "model_provider", "bedrock")
    monkeypatch.setattr(model_clients.settings, "bedrock_model_id", "")

    assert has_provider_key() is False


def test_create_chat_model_uses_demo_without_bedrock_model_id(monkeypatch):
    monkeypatch.setattr(model_clients.settings, "model_provider", "bedrock")
    monkeypatch.setattr(model_clients.settings, "bedrock_model_id", "")

    assert isinstance(create_chat_model(), DemoChatModel)


def test_bedrock_message_conversion_supports_system_text_and_images():
    encoded = base64.b64encode(b"fake-image").decode("ascii")

    system, messages = BedrockChatModel._convert_messages([
        SystemMessage(content="Use only the provided financial document."),
        HumanMessage(content=[
            {"type": "text", "text": "What does this table show?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
        ]),
    ])

    assert system == [{"text": "Use only the provided financial document."}]
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0] == {"text": "What does this table show?"}
    assert messages[0]["content"][1] == {
        "image": {"format": "jpeg", "source": {"bytes": b"fake-image"}}
    }


def test_bedrock_extracts_text_from_converse_response():
    response = {
        "output": {
            "message": {
                "content": [
                    {"text": "The liquidity coverage ratio was 113%."},
                    {"text": "Source: jpmorgan_2023_annual_report.pdf"},
                ]
            }
        }
    }

    assert BedrockChatModel._extract_text(response) == (
        "The liquidity coverage ratio was 113%.\nSource: jpmorgan_2023_annual_report.pdf"
    )

@pytest.mark.asyncio
async def test_resilient_chat_model_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(model_clients.settings, "model_call_max_retries", 1)
    monkeypatch.setattr(model_clients.settings, "model_call_timeout_seconds", 1.0)
    monkeypatch.setattr(model_clients.settings, "model_call_fallback_to_demo", True)

    class FlakyModel:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary provider outage")
            return AIMessage(content="provider recovered")

    primary = FlakyModel()
    model = ResilientChatModel(primary, fallback=DemoChatModel())

    response = await model.ainvoke([HumanMessage(content="hello")])

    assert response.content == "provider recovered"
    assert primary.calls == 2


@pytest.mark.asyncio
async def test_resilient_chat_model_falls_back_to_demo_after_retries(monkeypatch):
    monkeypatch.setattr(model_clients.settings, "model_call_max_retries", 0)
    monkeypatch.setattr(model_clients.settings, "model_call_timeout_seconds", 1.0)
    monkeypatch.setattr(model_clients.settings, "model_call_fallback_to_demo", True)

    class FailingModel:
        async def ainvoke(self, messages):
            raise RuntimeError("provider unavailable")

    model = ResilientChatModel(FailingModel(), fallback=DemoChatModel())

    response = await model.ainvoke([HumanMessage(content="hello")])

    assert "demo model" in str(response.content).lower()


@pytest.mark.asyncio
async def test_resilient_chat_model_raises_when_fallback_disabled(monkeypatch):
    monkeypatch.setattr(model_clients.settings, "model_call_max_retries", 0)
    monkeypatch.setattr(model_clients.settings, "model_call_timeout_seconds", 1.0)
    monkeypatch.setattr(model_clients.settings, "model_call_fallback_to_demo", False)

    class FailingModel:
        async def ainvoke(self, messages):
            raise RuntimeError("provider unavailable")

    model = ResilientChatModel(FailingModel(), fallback=DemoChatModel())

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await model.ainvoke([HumanMessage(content="hello")])
