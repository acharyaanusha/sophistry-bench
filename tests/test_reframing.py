import pytest

from sophistry_bench.rubric.reframing import score_reframing
from sophistry_bench.environment import Trajectory, DebaterTurn, JudgeRuling
from sophistry_bench.parser import ParsedTurn
from sophistry_bench.dataset import DebateTask
from sophistry_bench.agents import LLMClient


class _ConstJudge:
    def __init__(self, value: str):
        self.value = value

    async def chat_completion(self, *, messages, model, **kwargs):
        return self.value


def _traj() -> Trajectory:
    task = DebateTask(
        article_id="x", article="t", question="What color was the dress?",
        options=["red", "blue"], gold_index=0, round=1,
        debater_a_answer="red", debater_b_answer="blue",
    )
    turns = [
        DebaterTurn(debater="A", text="A faithfully argues red.",
                    parsed=ParsedTurn(claims=[], citations=[], raw="A")),
        DebaterTurn(debater="B", text="B reframes to symbolic meaning.",
                    parsed=ParsedTurn(claims=[], citations=[], raw="B")),
    ]
    return Trajectory(task=task, turns=turns, ruling=JudgeRuling(winner="A", reasoning=""))


@pytest.mark.asyncio
async def test_judge_high_means_no_reframing():
    judge = LLMClient(provider="openai", _override_client=_ConstJudge("0.95"))
    scores = await score_reframing(_traj(), judge_client=judge, judge_model="m")
    assert scores["A"] == pytest.approx(0.95)
    assert scores["B"] == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_judge_low_means_heavy_reframing():
    judge = LLMClient(provider="openai", _override_client=_ConstJudge("0.1"))
    scores = await score_reframing(_traj(), judge_client=judge, judge_model="m")
    assert scores["A"] == pytest.approx(0.1)
    assert scores["B"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_no_turns_returns_one():
    task = DebateTask(article_id="x", article="t", question="q",
                     options=["a","b"], gold_index=0, round=1,
                     debater_a_answer="a", debater_b_answer="b")
    judge = LLMClient(provider="openai", _override_client=_ConstJudge("0.0"))
    traj = Trajectory(task=task, turns=[], ruling=JudgeRuling(winner="A", reasoning=""))
    scores = await score_reframing(traj, judge_client=judge, judge_model="m")
    assert scores["A"] == 1.0
    assert scores["B"] == 1.0
