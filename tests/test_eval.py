import json

import pytest

from sophistry_bench.eval import EvalResult, evaluate_model
from sophistry_bench.dataset import DebateTask


def _tasks(n: int = 2) -> list[DebateTask]:
    return [
        DebateTask(
            article_id=f"a{i}", article="The summit was Tuesday.", question="When?",
            options=["Mon","Tue","Wed"], gold_index=1, round=1,
            debater_a_answer="Tue", debater_b_answer="Wed",
        )
        for i in range(n)
    ]


class _AlwaysReturns:
    def __init__(self, value: str):
        self.value = value
    async def chat_completion(self, *, messages, model, **kwargs):
        return self.value


@pytest.mark.asyncio
async def test_evaluate_model_returns_aggregated_subscores():
    from sophistry_bench.agents import LLMClient
    from sophistry_bench.environment import DebateEnv
    from sophistry_bench.rubric import JudgePool, SophistryRubric

    debater_a = LLMClient(provider="openai", _override_client=_AlwaysReturns("<claim>x</claim>"))
    debater_b = LLMClient(provider="openai", _override_client=_AlwaysReturns("<claim>y</claim>"))
    judge = LLMClient(provider="openai", _override_client=_AlwaysReturns("A"))
    env = DebateEnv(
        debater_a_client=debater_a, debater_a_model="m",
        debater_b_client=debater_b, debater_b_model="m",
        judge_client=judge, judge_model="m",
        turns_per_debater=2,
    )
    pool = JudgePool([("openai", "gpt-4o-mini", _AlwaysReturns("0.7"))])
    rubric = SophistryRubric(judge_pool=pool)

    result = await evaluate_model(env=env, rubric=rubric, tasks=_tasks(n=2))
    assert isinstance(result, EvalResult)
    assert result.n == 2
    assert "correctness" in result.mean_subscores
    assert 0.0 <= result.mean_subscores["aggregate"] <= 1.0
    assert len(result.trajectories) == 2


@pytest.mark.asyncio
async def test_run_leaderboard_writes_json(tmp_path):
    from sophistry_bench.eval import run_leaderboard

    out = tmp_path / "lb.json"
    debater_overrides = {
        "openai:gpt-4o-mini": _AlwaysReturns("<claim>x</claim>"),
        "openai:gpt-4o": _AlwaysReturns("<claim>y</claim>"),
    }
    judge_override = _AlwaysReturns("A")
    judge_pool_override = [_AlwaysReturns("0.5")]

    await run_leaderboard(
        debater_specs=[("openai", "gpt-4o-mini"), ("openai", "gpt-4o")],
        judge_spec=("openai", "gpt-4o-mini"),
        tasks=_tasks(n=1),
        output_path=out,
        turns_per_debater=2,
        debater_overrides=debater_overrides,
        judge_override=judge_override,
        judge_pool_overrides=judge_pool_override,
    )
    data = json.loads(out.read_text())
    assert "openai:gpt-4o-mini" in data
    assert "openai:gpt-4o" in data
    assert "mean_subscores" in data["openai:gpt-4o-mini"]


def test_compare_leaderboards_returns_deltas():
    from sophistry_bench.eval import compare_leaderboards
    before = {"openai:gpt-4o-mini": {"n": 10, "mean_subscores": {"aggregate": 0.5, "correctness": 0.6, "citation_bluffing": 0.7}}}
    after = {"openai:gpt-4o-mini-ft": {"n": 10, "mean_subscores": {"aggregate": 0.65, "correctness": 0.7, "citation_bluffing": 0.85}}}
    deltas = compare_leaderboards(before, after)
    assert deltas["aggregate"] == pytest.approx(0.15)
    assert deltas["citation_bluffing"] == pytest.approx(0.15)


def test_compare_leaderboards_warns_on_mismatched_keys():
    import warnings
    from sophistry_bench.eval import compare_leaderboards
    before = {"a:m": {"n": 1, "mean_subscores": {"aggregate": 0.5}}}
    after = {"b:m": {"n": 1, "mean_subscores": {"aggregate": 0.6}}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compare_leaderboards(before, after)
    assert any("different model keys" in str(w.message) for w in caught)
