import argparse
import asyncio
import json
from pathlib import Path

from sophistry_bench.dataset import build_debate_tasks, load_quality_from_json
from sophistry_bench.eval import run_leaderboard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("leaderboard.json"))
    parser.add_argument("--debaters", nargs="+", required=True,
                        help="Specs as provider:model, e.g. openai:gpt-4o")
    parser.add_argument("--judge", default="openai:gpt-4o-mini")
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--turns", type=int, default=3)
    args = parser.parse_args()

    items = load_quality_from_json(args.quality_json)[: args.n_tasks]
    tasks = []
    for item in items:
        distractor_idx = next(i for i in range(len(item.options)) if i != item.gold_index)
        tasks.extend(build_debate_tasks(item, distractor_index=distractor_idx))

    debater_specs = [tuple(d.split(":", 1)) for d in args.debaters]
    judge_spec = tuple(args.judge.split(":", 1))

    asyncio.run(run_leaderboard(
        debater_specs=debater_specs,
        judge_spec=judge_spec,
        tasks=tasks,
        output_path=args.output,
        turns_per_debater=args.turns,
    ))
    print(f"Wrote {args.output}")
    print(json.dumps(json.loads(args.output.read_text()), indent=2))


if __name__ == "__main__":
    main()
