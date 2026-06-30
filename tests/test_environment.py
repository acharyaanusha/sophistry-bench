import pytest

from sophistry_bench.environment import DebateEnv, Trajectory
from sophistry_bench.dataset import DebateTask
from sophistry_bench.agents import LLMClient


class _ScriptedClient:
    """Test backend that returns each canned response in order, one per call."""
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def chat_completion(self, *, messages, model, **kwargs):
        self.calls.append({"messages": messages, "model": model})
        return self._responses.pop(0)


def _task() -> DebateTask:
    return DebateTask(
        article_id="x", article="The summit was Tuesday.", question="When?",
        options=["Mon","Tue","Wed"], gold_index=1, round=1,
        debater_a_answer="Tue", debater_b_answer="Wed",
    )


@pytest.mark.asyncio
async def test_rollout_produces_expected_turn_count():
    debater_responses = [f"<claim>turn {i}</claim>" for i in range(6)]
    judge_responses = ["A"]
    debater_a = LLMClient(provider="openai", _override_client=_ScriptedClient(debater_responses[:3]))
    debater_b = LLMClient(provider="openai", _override_client=_ScriptedClient(debater_responses[3:]))
    judge = LLMClient(provider="openai", _override_client=_ScriptedClient(judge_responses))
    env = DebateEnv(
        debater_a_client=debater_a, debater_a_model="m",
        debater_b_client=debater_b, debater_b_model="m",
        judge_client=judge, judge_model="m",
        turns_per_debater=3,
    )
    traj = await env.rollout(_task())
    assert isinstance(traj, Trajectory)
    assert len(traj.turns) == 6  # 3 turns × 2 debaters
    assert traj.ruling is not None
    assert traj.ruling.winner == "A"


@pytest.mark.asyncio
async def test_rollout_alternates_debaters():
    debater_a = LLMClient(provider="openai", _override_client=_ScriptedClient(["A1","A2","A3"]))
    debater_b = LLMClient(provider="openai", _override_client=_ScriptedClient(["B1","B2","B3"]))
    judge = LLMClient(provider="openai", _override_client=_ScriptedClient(["B"]))
    env = DebateEnv(
        debater_a_client=debater_a, debater_a_model="m",
        debater_b_client=debater_b, debater_b_model="m",
        judge_client=judge, judge_model="m",
        turns_per_debater=3,
    )
    traj = await env.rollout(_task())
    assert [t.debater for t in traj.turns] == ["A","B","A","B","A","B"]


def test_load_environment_returns_callable_env():
    from sophistry_bench.environment import load_environment
    env = load_environment(config={
        "debater_a": {"provider": "openai", "model": "gpt-4o-mini"},
        "debater_b": {"provider": "openai", "model": "gpt-4o-mini"},
        "judge": {"provider": "openai", "model": "gpt-4o-mini"},
        "turns_per_debater": 2,
    })
    assert hasattr(env, "rollout")
    assert callable(env.rollout)


# ---------------------------------------------------------------------------
# vf_env.load_environment — debater_a / debater_b config
# ---------------------------------------------------------------------------

def _vf_env():
    from sophistry_bench import vf_env
    return vf_env


def _packaged_json() -> str:
    from sophistry_bench.dataset import packaged_quality_path
    return str(packaged_quality_path())


def test_vf_load_environment_self_play_uses_debater_for_both_sides():
    """With no debater_a/debater_b, both sides inherit the debater string."""
    env = _vf_env().load_environment(
        quality_json=_packaged_json(), n_items=2,
        debater="openai:gpt-4o-mini", judge="openai:gpt-4o-mini",
    )
    assert env._debate_env.a_model == "gpt-4o-mini"
    assert env._debate_env.b_model == "gpt-4o-mini"


def test_vf_load_environment_separate_debater_specs():
    """debater_a and debater_b configure independent models."""
    env = _vf_env().load_environment(
        quality_json=_packaged_json(), n_items=2,
        debater_a="openai:gpt-4o", debater_b="anthropic:claude-haiku-4-5",
        judge="openai:gpt-4o-mini",
    )
    assert env._debate_env.a_model == "gpt-4o"
    assert env._debate_env.b_model == "claude-haiku-4-5"


def test_vf_load_environment_partial_override_falls_back_to_debater():
    """Setting only debater_a leaves debater_b using the debater fallback."""
    env = _vf_env().load_environment(
        quality_json=_packaged_json(), n_items=2,
        debater="openai:gpt-4o-mini", debater_a="anthropic:claude-sonnet-4-6",
        judge="openai:gpt-4o-mini",
    )
    assert env._debate_env.a_model == "claude-sonnet-4-6"
    assert env._debate_env.b_model == "gpt-4o-mini"


def test_vf_load_environment_stores_debater_specs_on_env():
    """_debater_a_spec and _debater_b_spec are accessible for the warning."""
    env = _vf_env().load_environment(
        quality_json=_packaged_json(), n_items=2,
        debater_a="openai:gpt-4o", debater_b="anthropic:claude-haiku-4-5",
        judge="openai:gpt-4o-mini",
    )
    assert env._debater_a_spec == "openai:gpt-4o"
    assert env._debater_b_spec == "anthropic:claude-haiku-4-5"


def test_vf_load_environment_eval_split_is_set():
    """eval_dataset is populated by default (eval_fraction=0.1)."""
    env = _vf_env().load_environment(
        quality_json=_packaged_json(),
        debater="openai:gpt-4o-mini", judge="openai:gpt-4o-mini",
    )
    assert env.eval_dataset is not None
    assert len(env.eval_dataset) > 0
    # train + eval must account for all items; each item produces 2 rows
    total_rows = len(env.get_dataset()) + len(env.eval_dataset)
    assert total_rows % 2 == 0
    assert len(env.eval_dataset) == pytest.approx(total_rows * 0.1, abs=2)


def test_vf_load_environment_eval_fraction_zero_disables_split():
    """eval_fraction=0.0 means no eval split; eval_dataset stays None."""
    env = _vf_env().load_environment(
        quality_json=_packaged_json(), n_items=5,
        debater="openai:gpt-4o-mini", judge="openai:gpt-4o-mini",
        eval_fraction=0.0,
    )
    assert env.eval_dataset is None


def test_vf_load_environment_rollout_warns_on_framework_model(caplog):
    """A warning is emitted when the framework passes a model that will be ignored."""
    import logging
    env = _vf_env().load_environment(
        quality_json=_packaged_json(), n_items=2,
        debater_a="openai:gpt-4o", debater_b="anthropic:claude-haiku-4-5",
        judge="openai:gpt-4o-mini",
    )
    import asyncio
    fake_input = {"prompt": [{"role": "user", "content": "q"}], "answer": "a",
                  "task": "t", "info": {}, "example_id": 0}

    async def _run():
        # Pass a framework model; rollout will raise before LLM calls but the
        # warning fires before any I/O.
        try:
            await env.rollout(fake_input, None, "some-framework-model")
        except Exception:
            pass

    with caplog.at_level(logging.WARNING, logger="sophistry_bench.vf_env"):
        asyncio.run(_run())

    assert any("some-framework-model" in r.message for r in caplog.records)
    assert any("gpt-4o" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Gap 4: GRPO / trainee mode
# ---------------------------------------------------------------------------

def _make_chat_completion(text: str):
    """Build a minimal real openai.types.chat.ChatCompletion for tests."""
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    import time
    return ChatCompletion(
        id="test-id",
        object="chat.completion",
        created=int(time.time()),
        model="test-model",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=text),
                finish_reason="stop",
                logprobs=None,
            )
        ],
    )


class _FakeVLLMClient:
    """Minimal vLLM-compatible client stub that returns a real ChatCompletion."""
    def __init__(self, text: str):
        self._text = text
        self.calls: list[dict] = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    async def create(self, *, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return _make_chat_completion(self._text)


@pytest.mark.asyncio
async def test_grpo_trainee_a_uses_framework_client():
    """With trainee='A', the framework client is used for debater A's turns."""
    from sophistry_bench.dataset import DebateTask
    from sophistry_bench.agents import LLMClient
    from sophistry_bench.environment import DebateEnv
    from sophistry_bench.rubric import JudgePool, SophistryRubric
    from datasets import Dataset

    # Scripted opponents and judge via our internal clients
    class _AlwaysReturns:
        def __init__(self, v): self.v = v
        async def chat_completion(self, *, messages, model, **kw): return self.v

    b_client = LLMClient(provider="openai", _override_client=_AlwaysReturns("<claim>b</claim>"))
    judge_client = LLMClient(provider="openai", _override_client=_AlwaysReturns("A"))
    debate_env = DebateEnv(
        debater_a_client=LLMClient(provider="openai", _override_client=_AlwaysReturns("<claim>a</claim>")),
        debater_a_model="gpt-4o",
        debater_b_client=b_client,
        debater_b_model="gpt-4o-mini",
        judge_client=judge_client,
        judge_model="gpt-4o-mini",
        turns_per_debater=1,
    )

    task = DebateTask(
        article_id="x", article="The summit was Tuesday.", question="When?",
        options=["Mon", "Tue", "Wed"], gold_index=1, round=1,
        debater_a_answer="Tue", debater_b_answer="Wed",
    )
    info_dict = task.__dict__

    pool = JudgePool([("openai", "gpt-4o-mini", _AlwaysReturns("0.7"))])
    rubric = SophistryRubric(judge_pool=pool)
    ds = Dataset.from_list([{"prompt": [], "answer": "Tue", "task": "t",
                              "info": info_dict, "example_id": 0}])

    from sophistry_bench.vf_env import SophistryDebateEnv
    vllm_client = _FakeVLLMClient("<claim>trainee says Tue</claim>")
    env = SophistryDebateEnv(
        debate_env=debate_env,
        rubric_obj=rubric,
        dataset=ds,
        trainee="A",
    )

    fake_input = {"prompt": [], "answer": "Tue", "task": "t",
                  "info": info_dict, "example_id": 0}
    state = await env.rollout(fake_input, vllm_client, "trainee-model-v1")

    # Framework client was called for trainee A's turn
    assert len(vllm_client.calls) == 1
    assert vllm_client.calls[0]["model"] == "trainee-model-v1"
    # Trainee turn was threaded into trajectory
    assert len(state["trajectory"]) == 1
    # trajectory_id is present and non-empty
    assert state["trajectory_id"]
