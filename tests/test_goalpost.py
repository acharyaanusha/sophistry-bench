import pytest

from sophistry_bench.rubric.goalpost import score_goalpost
from sophistry_bench.environment import Trajectory, DebaterTurn, JudgeRuling
from sophistry_bench.parser import ParsedTurn
from sophistry_bench.dataset import DebateTask
from sophistry_bench.agents import LLMClient


class _ConstJudge:
    def __init__(self, value: str):
        self.value = value

    async def chat_completion(self, *, messages, model, **kwargs):
        return self.value


def _traj_with_turns(texts: list[str], winner: str = "A") -> Trajectory:
    task = DebateTask(
        article_id="x", article="text", question="q",
        options=["a","b"], gold_index=0, round=1,
        debater_a_answer="a", debater_b_answer="b",
    )
    turns = [
        DebaterTurn(debater="A", text=t, parsed=ParsedTurn(claims=[], citations=[], raw=t))
        for t in texts
    ]
    return Trajectory(task=task, turns=turns, ruling=JudgeRuling(winner=winner, reasoning=""))


@pytest.mark.asyncio
async def test_consistent_position_scores_high():
    judge = LLMClient(provider="openai", _override_client=_ConstJudge("0.95"))
    traj = _traj_with_turns(["First argument", "Second supporting argument", "Third"])
    assert await score_goalpost(traj, judge_client=judge, judge_model="gpt-4o-mini") == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_single_turn_returns_one():
    judge = LLMClient(provider="openai", _override_client=_ConstJudge("0.0"))
    traj = _traj_with_turns(["only turn"])
    assert await score_goalpost(traj, judge_client=judge, judge_model="x") == 1.0
