"""Multi-agent debate environment for Prime Intellect OpenEnv.

Entry point for ``prime eval run anusha/sophistry-bench`` and
``prime env install``.  Wraps the full sophistry-bench multi-agent debate
environment: two LLM debaters argue opposite sides of a QuALITY
reading-comprehension question; a weaker judge picks the winner.

Unlike the self-contained sprint env (which vendors its source), this env
depends on the main ``sophistry-bench`` package and re-exports its public API.
Install with::

    pip install sophistry-bench-env

or let ``prime env install`` handle it.

Usage modes
-----------
Eval / inference (default)::

    env = load_environment(
        debater_a="openai:gpt-4o",
        debater_b="anthropic:claude-haiku-4-5",
        judge="openai:gpt-4o-mini",
    )

GRPO training (trainee="A" uses the framework's vLLM client)::

    env = load_environment(
        debater="openai:gpt-4o-mini",      # opponent
        judge="openai:gpt-4o-mini",
        trainee="A",                        # vLLM client drives debater A
    )

Heterogeneous matchup evaluation::

    from sophistry_bench_env import run_leaderboard
    asyncio.run(run_leaderboard(
        matchups=[(("openai", "gpt-4o"), ("anthropic", "claude-sonnet-4-6"))],
        judge_spec=("openai", "gpt-4o-mini"),
        tasks=tasks,
        output_path=Path("results.json"),
    ))
"""

from __future__ import annotations

# Re-export the full public API so the OpenEnv server can access all symbols
# via ``import sophistry_bench_env`` without touching the main package directly.
from sophistry_bench.vf_env import load_environment  # noqa: F401
from sophistry_bench.eval import (  # noqa: F401
    run_leaderboard,
    evaluate_model,
    compare_leaderboards,
    EvalResult,
)
from sophistry_bench.environment import DebateEnv, Trajectory  # noqa: F401
from sophistry_bench.rubric import SophistryRubric, JudgePool  # noqa: F401
from sophistry_bench.agents import LLMClient  # noqa: F401
from sophistry_bench.dataset import (  # noqa: F401
    DebateTask,
    load_quality_from_json,
    packaged_quality_path,
)

__all__ = [
    # Primary entry point
    "load_environment",
    # Evaluation helpers
    "run_leaderboard",
    "evaluate_model",
    "compare_leaderboards",
    "EvalResult",
    # Core types (needed for scripted test harnesses)
    "DebateEnv",
    "Trajectory",
    "SophistryRubric",
    "JudgePool",
    "LLMClient",
    "DebateTask",
    # Dataset utilities
    "load_quality_from_json",
    "packaged_quality_path",
]
