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
