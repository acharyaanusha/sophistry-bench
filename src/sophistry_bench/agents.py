from dataclasses import dataclass
from typing import Literal, Protocol

from anthropic import AsyncAnthropic, Omit
from google import genai
from google.genai.types import GenerateContentConfig
from openai import AsyncOpenAI

Provider = Literal["openai", "anthropic", "google"]


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class _ChatBackend(Protocol):
    async def chat_completion(self, *, messages: list[dict], model: str, **kwargs) -> str: ...


class _OpenAIBackend:
    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client

    async def chat_completion(self, *, messages: list[dict], model: str, **kwargs) -> str:
        resp = await self._get_client().chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        return resp.choices[0].message.content or ""


class _AnthropicBackend:
    def __init__(self) -> None:
        self._client: AsyncAnthropic | None = None

    def _get_client(self) -> AsyncAnthropic:
        if self._client is None:
            self._client = AsyncAnthropic()
        return self._client

    async def chat_completion(self, *, messages: list[dict], model: str, **kwargs) -> str:
        system_messages = [m["content"] for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]
        system: str | Omit = "\n".join(system_messages) if system_messages else Omit()
        kwargs.setdefault("max_tokens", 2048)
        resp = await self._get_client().messages.create(
            model=model,
            messages=non_system,
            system=system,
            **kwargs,
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class _GoogleBackend:
    def __init__(self) -> None:
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client()
        return self._client

    async def chat_completion(self, *, messages: list[dict], model: str, **kwargs) -> str:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        config = GenerateContentConfig(**kwargs) if kwargs else None
        resp = await self._get_client().aio.models.generate_content(
            model=model, contents=prompt, config=config
        )
        return resp.text or ""


class LLMClient:
    def __init__(self, provider: Provider, *, _override_client: _ChatBackend | None = None):
        if _override_client is not None:
            self._backend: _ChatBackend = _override_client
            return
        if provider == "openai":
            self._backend = _OpenAIBackend()
        elif provider == "anthropic":
            self._backend = _AnthropicBackend()
        elif provider == "google":
            self._backend = _GoogleBackend()
        else:
            raise ValueError(f"unknown provider: {provider}")

    async def generate(
        self, *, messages: list[Message], model: str, temperature: float = 0.7, **kwargs
    ) -> str:
        return await self._backend.chat_completion(
            messages=[m.to_dict() for m in messages],
            model=model,
            temperature=temperature,
            **kwargs,
        )
