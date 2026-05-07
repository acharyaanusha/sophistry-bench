import pytest

from sophistry_bench.rubric.aggregate import SophistryRubric, JudgePool
from sophistry_bench.environment import Trajectory, DebaterTurn, JudgeRuling
from sophistry_bench.parser import ParsedTurn
from sophistry_bench.dataset import DebateTask
from sophistry_bench.agents import LLMClient


class _ConstJudge:
    def __init__(self, value: str):
        self.value = value

    async def chat_completion(self, *, messages, model, **kwargs):
        return self.value


def _trajectory():
    task = DebateTask(
        article_id="x", article="The summit was reached Tuesday.", question="q",
        options=["Mon","Tue","Wed"], gold_index=1, round=1,
        debater_a_answer="Tue", debater_b_answer="Wed",
    )
    turn = DebaterTurn(
        debater="A", text="I argue Tuesday.",
        parsed=ParsedTurn(claims=["The summit was reached Tuesday."], citations=["The summit was reached Tuesday."], raw=""),
    )
    return Trajectory(task=task, turns=[turn], ruling=JudgeRuling(winner="A", reasoning=""))


@pytest.mark.asyncio
async def test_rubric_returns_all_subscores_and_aggregate():
    pool = JudgePool([
        ("openai", "gpt-4o-mini", _ConstJudge("0.8")),
        ("openai", "gpt-4o-mini", _ConstJudge("0.9")),
        ("openai", "gpt-4o-mini", _ConstJudge("0.7")),
    ])
    rubric = SophistryRubric(judge_pool=pool)
    scores = await rubric.score(_trajectory())
    assert "correctness" in scores
    assert "citation_bluffing" in scores
    assert "sycophantic" in scores
    assert "false_confidence" in scores
    assert "gish_gallop" in scores
    assert "goalpost" in scores
    assert "aggregate" in scores
    assert all(0.0 <= v <= 1.0 for v in scores.values())


@pytest.mark.asyncio
async def test_aggregate_uses_median_of_judges():
    pool = JudgePool([
        ("openai", "gpt-4o-mini", _ConstJudge("0.2")),
        ("openai", "gpt-4o-mini", _ConstJudge("0.5")),
        ("openai", "gpt-4o-mini", _ConstJudge("0.9")),
    ])
    rubric = SophistryRubric(judge_pool=pool)
    scores = await rubric.score(_trajectory())
    # median of 0.2, 0.5, 0.9 is 0.5 — used in LLM-judge sub-rubrics
    assert abs(scores["false_confidence"] - 0.5) < 0.01
