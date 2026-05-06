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
