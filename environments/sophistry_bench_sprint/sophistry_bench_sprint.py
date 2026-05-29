"""Hub entry point for the sophistry-bench Reward Hacking Sprint env.

Thin wrapper that re-exports ``load_environment`` from ``sophistry_bench.sprint``.
The actual env implementation, unit tests, and docs all live in the main
sophistry-bench package — see
https://github.com/acharyaanusha/sophistry-bench for source,
docs/reward-hacking.md for the pre-registered hypothesis, and
src/sophistry_bench/sprint/env.py for the env code.

Keeping a single source of truth (the main package) avoids the code-drift
risk of maintaining a parallel implementation in this Hub package.
"""

from sophistry_bench.sprint import load_environment

__all__ = ["load_environment"]
