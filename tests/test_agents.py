import pytest
from sophistry_bench.agents import LLMClient, Message
from tests.fixtures.mock_clients import MockOpenAIClient


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
        LLMClient(provider="not-a-provider")
