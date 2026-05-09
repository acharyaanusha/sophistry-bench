import pytest

from sophistry_bench.rubric.aggregate import SophistryRubric, JudgePool
from sophistry_bench.environment import Trajectory, DebaterTurn, JudgeRuling
from sophistry_bench.parser import ParsedTurn
from sophistry_bench.dataset import DebateTask


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
    turns = [
        DebaterTurn(debater="A", text="I argue Tuesday.",
                    parsed=ParsedTurn(claims=["Tuesday."], citations=["The summit was reached Tuesday."], raw="")),
        DebaterTurn(debater="B", text="I argue Wednesday.",
                    parsed=ParsedTurn(claims=["wrong claim"], citations=[], raw="")),
    ]
    return Trajectory(task=task, turns=turns, ruling=JudgeRuling(winner="A", reasoning=""))


@pytest.mark.asyncio
async def test_rubric_returns_all_axes_with_per_side_keys():
    pool = JudgePool([
        ("openai", "gpt-4o-mini", _ConstJudge("0.8")),
        ("openai", "gpt-4o-mini", _ConstJudge("0.9")),
        ("openai", "gpt-4o-mini", _ConstJudge("0.7")),
    ])
    rubric = SophistryRubric(judge_pool=pool)
    scores = await rubric.score(_trajectory())

    for axis in ("correctness", "citation_bluffing", "sycophantic",
                 "false_confidence", "gish_gallop", "goalpost", "reframing"):
        assert axis in scores
        assert f"{axis}_A" in scores
        assert f"{axis}_B" in scores
    assert "aggregate" in scores
    # values are bounded (per-side keys may be 0 or 1; mean stays in [0,1])
    for k, v in scores.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"


@pytest.mark.asyncio
async def test_aggregate_uses_median_per_key_across_judges():
    pool = JudgePool([
        ("openai", "gpt-4o-mini", _ConstJudge("0.2")),
        ("openai", "gpt-4o-mini", _ConstJudge("0.5")),
        ("openai", "gpt-4o-mini", _ConstJudge("0.9")),
    ])
    rubric = SophistryRubric(judge_pool=pool)
    scores = await rubric.score(_trajectory())
    # Each LLM-judge axis returns the per-judge dict {A,B,mean}; median across
    # the 3 judges of each key equals 0.5 (since A=B=value for each judge).
    assert scores["false_confidence"] == pytest.approx(0.5, abs=0.01)
    assert scores["false_confidence_A"] == pytest.approx(0.5, abs=0.01)
    assert scores["false_confidence_B"] == pytest.approx(0.5, abs=0.01)
