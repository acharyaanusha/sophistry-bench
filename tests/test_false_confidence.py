import pytest

from sophistry_bench.rubric.false_confidence import score_false_confidence
from sophistry_bench.environment import Trajectory, DebaterTurn, JudgeRuling
from sophistry_bench.parser import ParsedTurn
from sophistry_bench.dataset import DebateTask
from sophistry_bench.agents import LLMClient


class _ConstJudge:
    """Test backend that always returns the same string from chat_completion."""
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
    turn = DebaterTurn(
        debater="A", text="I am 100% certain my answer is correct.",
        parsed=ParsedTurn(
            claims=["The summit was reached Tuesday with absolute certainty."],
            citations=[], raw="",
        ),
    )
    return Trajectory(task=task, turns=[turn], ruling=JudgeRuling(winner=winner, reasoning=""))


@pytest.mark.asyncio
async def test_judge_returns_low_when_confidence_decoupled_from_accuracy():
    judge = LLMClient(provider="openai", _override_client=_ConstJudge("0.2"))
    score = await score_false_confidence(_traj(), judge_client=judge, judge_model="gpt-4o-mini")
    assert score == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_judge_returns_high_when_confidence_aligned():
    judge = LLMClient(provider="openai", _override_client=_ConstJudge("0.95"))
    score = await score_false_confidence(_traj(), judge_client=judge, judge_model="gpt-4o-mini")
    assert score == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_judge_malformed_response_returns_neutral():
    judge = LLMClient(provider="openai", _override_client=_ConstJudge("I don't know how to score this"))
    score = await score_false_confidence(_traj(), judge_client=judge, judge_model="gpt-4o-mini")
    assert score == 0.5
