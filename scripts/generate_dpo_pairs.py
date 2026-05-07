import argparse
import asyncio
import json
from pathlib import Path

from sophistry_bench.agents import LLMClient
from sophistry_bench.dataset import build_debate_tasks, load_quality_from_json
from sophistry_bench.environment import DebateEnv
from sophistry_bench.eval import evaluate_model
from sophistry_bench.rubric import JudgePool, SophistryRubric
from sophistry_bench.train import build_dpo_pairs


async def _run(args: argparse.Namespace) -> None:
    items = load_quality_from_json(args.quality_json)[: args.n_items]
    tasks = []
    for item in items:
        distractor_idx = next(i for i in range(len(item.options)) if i != item.gold_index)
        tasks.extend(build_debate_tasks(item, distractor_index=distractor_idx))

    provider, model = args.debater.split(":", 1)
    j_provider, j_model = args.judge.split(":", 1)
    env = DebateEnv(
        debater_a_client=LLMClient(provider=provider), debater_a_model=model,
        debater_b_client=LLMClient(provider=provider), debater_b_model=model,
        judge_client=LLMClient(provider=j_provider), judge_model=j_model,
        turns_per_debater=args.turns,
    )
    pool = JudgePool([(j_provider, j_model, None) for _ in range(3)])
    rubric = SophistryRubric(judge_pool=pool)
    result = await evaluate_model(env=env, rubric=rubric, tasks=tasks)
    scores_by_id = {id(t): s for t, s in zip(result.trajectories, result.per_task_scores)}
    pairs = build_dpo_pairs(result.trajectories, scores_by_id, sophistry_threshold=args.threshold)
    Path(args.output).write_text("\n".join(json.dumps(p) for p in pairs))
    print(f"Wrote {len(pairs)} pairs to {args.output}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quality-json", type=Path, required=True)
    p.add_argument("--debater", default="openai:gpt-4o-mini")
    p.add_argument("--judge", default="openai:gpt-4o-mini")
    p.add_argument("--turns", type=int, default=3)
    p.add_argument("--n-items", type=int, default=500)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--output", type=Path, default=Path("dpo_pairs.jsonl"))
    args = p.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
