import argparse
import asyncio
import json
import os
from pathlib import Path

from sophistry_bench.dataset import build_debate_tasks, load_quality_from_json, pick_distractor, stable_hash
from sophistry_bench.eval import run_leaderboard


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("leaderboard.json"))
    parser.add_argument("--debaters", nargs="+", required=True,
                        help="Specs as provider:model, e.g. openai:gpt-4o or anthropic:claude-sonnet-4-6. "
                             "Per Khan et al. 2024, debaters should be stronger than the judge.")
    parser.add_argument("--judge", default="openai:gpt-4o-mini",
                        help="Judge spec as provider:model (default: openai:gpt-4o-mini). "
                             "Per Khan et al. 2024, should be a weaker model than the debaters "
                             "(non-expert-judge setup); e.g. anthropic:claude-haiku-4-5 "
                             "when debating with Anthropic models.")
    parser.add_argument("--n-tasks", type=int, default=20,
                        help="Number of QuALITY items (each yields 2 debate rounds)")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--judge-pool-size", type=int, default=3,
                        help="Number of judges (homogeneous pool from --judge); ignored if --judge-pool is given")
    parser.add_argument("--judge-pool", nargs="+", default=None,
                        help="Heterogeneous judge pool, e.g. anthropic:claude-haiku-4-5 anthropic:claude-sonnet-4-6")
    args = parser.parse_args()

    _load_env_file(Path(__file__).parent.parent / ".env")

    items = load_quality_from_json(args.quality_json)[: args.n_tasks]
    tasks = []
    for item in items:
        distractor_idx = pick_distractor(item, seed=stable_hash(item.article_id))
        tasks.extend(build_debate_tasks(item, distractor_index=distractor_idx))

    debater_specs = [tuple(d.split(":", 1)) for d in args.debaters]
    judge_spec = tuple(args.judge.split(":", 1))
    pool_specs = [tuple(j.split(":", 1)) for j in args.judge_pool] if args.judge_pool else None

    asyncio.run(run_leaderboard(
        debater_specs=debater_specs,
        judge_spec=judge_spec,
        tasks=tasks,
        output_path=args.output,
        turns_per_debater=args.turns,
        judge_pool_size=args.judge_pool_size,
        pool_specs=pool_specs,
    ))
    print(f"Wrote {args.output}")
    print(json.dumps(json.loads(args.output.read_text()), indent=2))


if __name__ == "__main__":
    main()
