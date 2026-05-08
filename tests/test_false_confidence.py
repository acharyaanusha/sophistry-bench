import pytest

from sophistry_bench.rubric.false_confidence import score_false_confidence
from sophistry_bench.environment import Trajectory, DebaterTurn, JudgeRuling
from sophistry_bench.parser import ParsedTurn
from sophistry_bench.dataset import DebateTask
from sophistry_bench.agents import LLMClient


class _ConstJudge:
    def __init__(self, value: str):
        self.value = value

    async def chat_completion(self, *, messages, model, **kwargs):
        return self.value


def _traj(winner="A"):
    task = DebateTask(
        article_id="x", article="The summit was reached Tuesday.", question="q",
        options=["Mon", "Tue", "Wed"], gold_index=1, round=1,
        debater_a_answer="Tue", debater_b_answer="Wed",
    )
    turns = [
        DebaterTurn(debater="A", text="A turn",
                    parsed=ParsedTurn(claims=["A claim"], citations=[], raw="")),
        DebaterTurn(debater="B", text="B turn",
                    parsed=ParsedTurn(claims=["B claim"], citations=[], raw="")),
    ]
    return Trajectory(task=task, turns=turns, ruling=JudgeRuling(winner=winner, reasoning=""))


@pytest.mark.asyncio
async def test_judge_low_returns_low_per_side_and_mean():
    judge = LLMClient(provider="openai", _override_client=_ConstJudge("0.2"))
    scores = await score_false_confidence(_traj(), judge_client=judge, judge_model="gpt-4o-mini")
    assert scores["A"] == pytest.approx(0.2)
    assert scores["B"] == pytest.approx(0.2)
    assert scores["mean"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_judge_high_returns_high_per_side_and_mean():
    judge = LLMClient(provider="openai", _override_client=_ConstJudge("0.95"))
    scores = await score_false_confidence(_traj(), judge_client=judge, judge_model="gpt-4o-mini")
    assert scores["A"] == pytest.approx(0.95)
    assert scores["B"] == pytest.approx(0.95)
    assert scores["mean"] == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_judge_malformed_returns_neutral():
    judge = LLMClient(provider="openai", _override_client=_ConstJudge("I don't know"))
    scores = await score_false_confidence(_traj(), judge_client=judge, judge_model="gpt-4o-mini")
    assert scores["A"] == 0.5
    assert scores["B"] == 0.5
    assert scores["mean"] == 0.5
