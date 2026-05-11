import asyncio

import pytest
from sophistry_bench.agents import LLMClient, Message
from tests.fixtures.mock_clients import MockOpenAIClient, MockAnthropicClient, MockGoogleClient


@pytest.mark.asyncio
async def test_openai_client_returns_completion():
    mock = MockOpenAIClient(canned={"hello": "hi there"})
    client = LLMClient(provider="openai", _override_client=mock)
    messages = [Message(role="user", content="hello")]
    result = await client.generate(messages=messages, model="gpt-4o-mini")
    assert result == "hi there"


@pytest.mark.asyncio
async def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        LLMClient(provider="not-a-provider")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_anthropic_client():
    mock = MockAnthropicClient(canned={"ping": "pong-anthropic"})
    client = LLMClient(provider="anthropic", _override_client=mock)
    result = await client.generate(
        messages=[Message(role="user", content="ping")], model="claude-haiku-4-5"
    )
    assert result == "pong-anthropic"


@pytest.mark.asyncio
async def test_google_client():
    mock = MockGoogleClient(canned={"ping": "pong-google"})
    client = LLMClient(provider="google", _override_client=mock)
    result = await client.generate(
        messages=[Message(role="user", content="ping")], model="gemini-2.5-flash"
    )
    assert result == "pong-google"


@pytest.mark.asyncio
async def test_anthropic_backend_routes_system_message():
    """System messages are concatenated and passed via the `system` param, not in messages."""
    from unittest.mock import AsyncMock, MagicMock
    from sophistry_bench.agents import _AnthropicBackend

    mock_sdk = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = "reply"
    mock_response = MagicMock()
    mock_response.content = [block]
    mock_sdk.messages.create = AsyncMock(return_value=mock_response)

    backend = _AnthropicBackend()
    backend._clients[asyncio.get_running_loop()] = mock_sdk
    result = await backend.chat_completion(
        messages=[
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "hi"},
        ],
        model="claude-haiku-4-5",
    )
    assert result == "reply"
    call_kwargs = mock_sdk.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "Be helpful"
    assert all(m["role"] != "system" for m in call_kwargs["messages"])


@pytest.mark.asyncio
async def test_anthropic_backend_omits_system_when_none():
    """When no system message, `system` should be Omit() (not empty string)."""
    from unittest.mock import AsyncMock, MagicMock
    from anthropic import Omit
    from sophistry_bench.agents import _AnthropicBackend

    mock_sdk = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = "ok"
    mock_response = MagicMock()
    mock_response.content = [block]
    mock_sdk.messages.create = AsyncMock(return_value=mock_response)

    backend = _AnthropicBackend()
    backend._clients[asyncio.get_running_loop()] = mock_sdk
    await backend.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        model="claude-haiku-4-5",
    )
    call_kwargs = mock_sdk.messages.create.call_args.kwargs
    assert isinstance(call_kwargs["system"], Omit)
