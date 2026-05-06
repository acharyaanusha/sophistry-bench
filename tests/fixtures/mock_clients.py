from dataclasses import dataclass, field


@dataclass
class MockOpenAIClient:
    canned: dict[str, str] = field(default_factory=dict)
    calls: list[dict] = field(default_factory=list)

    async def chat_completion(self, *, messages, model, **kwargs):
        self.calls.append({"messages": messages, "model": model, **kwargs})
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return self.canned.get(last_user, "MOCK RESPONSE")
