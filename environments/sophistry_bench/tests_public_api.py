"""Guards the public API the OpenEnv server depends on."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Make the env module importable when running tests from the env directory.
_ENV_DIR = Path(__file__).parent
if str(_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(_ENV_DIR))


def test_all_public_symbols_present():
    m = importlib.import_module("sophistry_bench_env")
    required = {
        "load_environment",
        "run_leaderboard",
        "evaluate_model",
        "compare_leaderboards",
        "EvalResult",
        "DebateEnv",
        "Trajectory",
        "SophistryRubric",
        "JudgePool",
        "LLMClient",
        "DebateTask",
        "load_quality_from_json",
        "packaged_quality_path",
    }
    missing = required - set(m.__all__)
    assert not missing, f"missing from __all__: {missing}"
    for name in required:
        assert hasattr(m, name), f"missing attribute: {name}"


def test_load_environment_returns_vf_environment():
    """load_environment() must return a verifiers Environment usable for eval."""
    import verifiers as vf
    from sophistry_bench_env import load_environment, packaged_quality_path

    env = load_environment(
        quality_json=str(packaged_quality_path()),
        n_items=2,
        eval_fraction=0.0,
        debater="openai:gpt-4o-mini",
        judge="openai:gpt-4o-mini",
    )
    assert isinstance(env, vf.Environment)
    assert callable(env.rollout)


def test_load_environment_trainee_a_sets_attribute():
    """trainee='A' is stored on the env and activates the GRPO path."""
    from sophistry_bench_env import load_environment, packaged_quality_path

    env = load_environment(
        quality_json=str(packaged_quality_path()),
        n_items=2,
        eval_fraction=0.0,
        debater="openai:gpt-4o-mini",
        judge="openai:gpt-4o-mini",
        trainee="A",
    )
    assert env._trainee == "A"


def test_load_environment_separate_debater_specs():
    """debater_a and debater_b configure independent models."""
    from sophistry_bench_env import load_environment, packaged_quality_path

    env = load_environment(
        quality_json=str(packaged_quality_path()),
        n_items=2,
        eval_fraction=0.0,
        debater_a="openai:gpt-4o",
        debater_b="anthropic:claude-haiku-4-5",
        judge="openai:gpt-4o-mini",
    )
    assert env._debate_env.a_model == "gpt-4o"
    assert env._debate_env.b_model == "claude-haiku-4-5"


def test_eval_fraction_default_creates_eval_split():
    """Default eval_fraction=0.1 produces a non-empty eval dataset."""
    from sophistry_bench_env import load_environment, packaged_quality_path

    env = load_environment(
        quality_json=str(packaged_quality_path()),
        debater="openai:gpt-4o-mini",
        judge="openai:gpt-4o-mini",
    )
    assert env.eval_dataset is not None
    assert len(env.eval_dataset) > 0


def test_compare_leaderboards_importable():
    """compare_leaderboards is re-exported and callable."""
    from sophistry_bench_env import compare_leaderboards

    before = {"openai:gpt-4o-mini": {"n": 5, "mean_subscores": {"aggregate": 0.5}}}
    after = {"openai:gpt-4o-mini": {"n": 5, "mean_subscores": {"aggregate": 0.65}}}
    deltas = compare_leaderboards(before, after)
    assert abs(deltas["aggregate"] - 0.15) < 1e-9
