import pytest

from sophistry_bench.rubric.gish_gallop import score_gish_gallop
from sophistry_bench.environment import Trajectory, DebaterTurn, JudgeRuling
from sophistry_bench.parser import ParsedTurn
from sophistry_bench.dataset import DebateTask
from sophistry_bench.agents import LLMClient


class _ConstJudge:
    def __init__(self, value: str):
        self.value = value

    async def chat_completion(self, *, messages, model, **kwargs):
        return self.value


def _traj(claim_count: int, winner: str = "A") -> Trajectory:
    task = DebateTask(
        article_id="x", article="text", question="q",
        options=["a", "b"], gold_index=0, round=1,
        debater_a_answer="a", debater_b_answer="b",
    )
    turn = DebaterTurn(
        debater="A", text="x",
        parsed=ParsedTurn(claims=[f"c{i}" for i in range(claim_count)], citations=[], raw=""),
    )
    return Trajectory(task=task, turns=[turn], ruling=JudgeRuling(winner=winner, reasoning=""))


@pytest.mark.asyncio
async def test_few_quality_claims_score_high():
    judge = LLMClient(provider="openai", _override_client=_ConstJudge("0.9"))
    score = await score_gish_gallop(_traj(claim_count=3), judge_client=judge, judge_model="gpt-4o-mini")
    assert score == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_many_low_quality_claims_score_low():
    judge = LLMClient(provider="openai", _override_client=_ConstJudge("0.3"))
    score = await score_gish_gallop(_traj(claim_count=15), judge_client=judge, judge_model="gpt-4o-mini")
    assert score < 0.2


@pytest.mark.asyncio
async def test_no_winner_returns_neutral():
    judge = LLMClient(provider="openai", _override_client=_ConstJudge("0.5"))
    task = DebateTask(article_id="x", article="t", question="q", options=["a","b"], gold_index=0,
                     round=1, debater_a_answer="a", debater_b_answer="b")
    traj = Trajectory(task=task, turns=[], ruling=None)
    assert await score_gish_gallop(traj, judge_client=judge, judge_model="x") == 0.5
