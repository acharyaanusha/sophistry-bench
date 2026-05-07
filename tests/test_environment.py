import pytest

from sophistry_bench.environment import DebateEnv, Trajectory
from sophistry_bench.dataset import DebateTask
from sophistry_bench.agents import LLMClient


class _ScriptedClient:
    """Test backend that returns each canned response in order, one per call."""
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def chat_completion(self, *, messages, model, **kwargs):
        self.calls.append({"messages": messages, "model": model})
        return self._responses.pop(0)


def _task() -> DebateTask:
    return DebateTask(
        article_id="x", article="The summit was Tuesday.", question="When?",
        options=["Mon","Tue","Wed"], gold_index=1, round=1,
        debater_a_answer="Tue", debater_b_answer="Wed",
    )


@pytest.mark.asyncio
async def test_rollout_produces_expected_turn_count():
    debater_responses = [f"<claim>turn {i}</claim>" for i in range(6)]
    judge_responses = ["A"]
    debater_a = LLMClient(provider="openai", _override_client=_ScriptedClient(debater_responses[:3]))
    debater_b = LLMClient(provider="openai", _override_client=_ScriptedClient(debater_responses[3:]))
    judge = LLMClient(provider="openai", _override_client=_ScriptedClient(judge_responses))
    env = DebateEnv(
        debater_a_client=debater_a, debater_a_model="m",
        debater_b_client=debater_b, debater_b_model="m",
        judge_client=judge, judge_model="m",
        turns_per_debater=3,
    )
    traj = await env.rollout(_task())
    assert isinstance(traj, Trajectory)
    assert len(traj.turns) == 6  # 3 turns × 2 debaters
    assert traj.ruling is not None
    assert traj.ruling.winner == "A"


@pytest.mark.asyncio
async def test_rollout_alternates_debaters():
    debater_a = LLMClient(provider="openai", _override_client=_ScriptedClient(["A1","A2","A3"]))
    debater_b = LLMClient(provider="openai", _override_client=_ScriptedClient(["B1","B2","B3"]))
    judge = LLMClient(provider="openai", _override_client=_ScriptedClient(["B"]))
    env = DebateEnv(
        debater_a_client=debater_a, debater_a_model="m",
        debater_b_client=debater_b, debater_b_model="m",
        judge_client=judge, judge_model="m",
        turns_per_debater=3,
    )
    traj = await env.rollout(_task())
    assert [t.debater for t in traj.turns] == ["A","B","A","B","A","B"]
