from dataclasses import dataclass
from typing import Literal, Protocol

from openai import AsyncOpenAI

Provider = Literal["openai"]


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


class LLMClient:
    def __init__(self, provider: Provider, *, _override_client: _ChatBackend | None = None):
        if _override_client is not None:
            self._backend: _ChatBackend = _override_client
            return
        if provider == "openai":
            self._backend = _OpenAIBackend()
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
