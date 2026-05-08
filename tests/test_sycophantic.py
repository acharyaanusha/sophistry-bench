import pytest
from sophistry_bench.rubric.sycophantic import score_sycophantic
from sophistry_bench.environment import Trajectory, DebaterTurn
from sophistry_bench.parser import ParsedTurn
from sophistry_bench.dataset import DebateTask
from sophistry_bench.agents import LLMClient


class _ConstJudge:
    """Mock backend that always returns a fixed string."""
    def __init__(self, value: str):
        self.value = value
    async def chat_completion(self, *, messages, model, **kwargs):
        return self.value


def _task() -> DebateTask:
    return DebateTask(
        article_id="x", article="text", question="q",
        options=["a", "b"], gold_index=0, round=1,
        debater_a_answer="a", debater_b_answer="b",
    )


def _turn(debater, text):
    return DebaterTurn(debater=debater, text=text, parsed=ParsedTurn(claims=[], citations=[], raw=text))


@pytest.mark.asyncio
async def test_high_score_when_judge_reports_no_shift():
    judge = LLMClient(provider="openai", _override_client=_ConstJudge("0.95"))
    traj = Trajectory(
        task=_task(),
        turns=[_turn("A", "I argue A."), _turn("A", "I still argue A.")],
    )
    scores = await score_sycophantic(traj, judge_client=judge, judge_model="gpt-4o-mini")
    assert scores["A"] == pytest.approx(0.95, abs=0.01)


@pytest.mark.asyncio
async def test_low_score_when_judge_reports_shift():
    judge = LLMClient(provider="openai", _override_client=_ConstJudge("0.10"))
    traj = Trajectory(
        task=_task(),
        turns=[_turn("A", "I argue A strongly."), _turn("A", "Perhaps the more nuanced reading is B.")],
    )
    scores = await score_sycophantic(traj, judge_client=judge, judge_model="gpt-4o-mini")
    assert scores["A"] == pytest.approx(0.10, abs=0.01)


@pytest.mark.asyncio
async def test_single_turn_returns_one():
    judge = LLMClient(provider="openai", _override_client=_ConstJudge("0.5"))
    traj = Trajectory(task=_task(), turns=[_turn("A", "Solo.")])
    scores = await score_sycophantic(traj, judge_client=judge, judge_model="gpt-4o-mini")
    assert scores["A"] == 1.0  # nothing to shift from


@pytest.mark.asyncio
async def test_per_side_independence():
    """Each side gets scored separately; the per-side scores should reflect each side's own turns."""
    judge = LLMClient(provider="openai", _override_client=_ConstJudge("0.7"))
    traj = Trajectory(
        task=_task(),
        turns=[
            _turn("A", "I argue A."), _turn("A", "Still A."),
            _turn("B", "I argue B."), _turn("B", "Still B."),
        ],
    )
    scores = await score_sycophantic(traj, judge_client=judge, judge_model="gpt-4o-mini")
    assert scores["A"] == pytest.approx(0.7, abs=0.01)
    assert scores["B"] == pytest.approx(0.7, abs=0.01)
    assert scores["mean"] == pytest.approx(0.7, abs=0.01)


@pytest.mark.asyncio
async def test_returns_mid_value_when_judge_output_unparseable():
    judge = LLMClient(provider="openai", _override_client=_ConstJudge("not a number"))
    traj = Trajectory(
        task=_task(),
        turns=[_turn("A", "I argue A."), _turn("A", "Still A.")],
    )
    scores = await score_sycophantic(traj, judge_client=judge, judge_model="gpt-4o-mini")
    assert scores["A"] == 0.5  # fallback when judge output can't be parsed
