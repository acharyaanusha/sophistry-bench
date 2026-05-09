"""Run a single debate with two different debater models head-to-head.

Useful for validating fine-tunes by qualitative comparison: pit the trained
model against the base on the same QuALITY item and inspect both transcripts
plus per-side rubric scores.

Example:
    python scripts/head_to_head.py \\
      --debater-a openai:gpt-4o-2024-08-06 \\
      --debater-b openai:ft:gpt-4o-2024-08-06:personal:sophistry-pol:DdiUviSD \\
      --judge anthropic:claude-haiku-4-5
"""
import argparse
import asyncio
import os
from pathlib import Path

from sophistry_bench.agents import LLMClient
from sophistry_bench.dataset import (
    build_debate_tasks,
    load_quality_from_json,
    pick_distractor,
    stable_hash,
)
from sophistry_bench.environment import DebateEnv
from sophistry_bench.rubric import JudgePool, SophistryRubric


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-json", default="data/quality_dev.json")
    parser.add_argument("--debater-a", required=True, help="provider:model — argues the GOLD answer")
    parser.add_argument("--debater-b", required=True, help="provider:model — argues a DISTRACTOR")
    parser.add_argument("--judge", default="anthropic:claude-haiku-4-5")
    parser.add_argument("--item-index", type=int, default=0)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--judge-pool-size", type=int, default=3)
    args = parser.parse_args()

    _load_env_file(Path(__file__).parent.parent / ".env")

    items = load_quality_from_json(Path(args.quality_json))
    item = items[args.item_index]
    distractor_idx = pick_distractor(item, seed=stable_hash(item.article_id))
    task = build_debate_tasks(item, distractor_index=distractor_idx)[0]

    a_provider, a_model = args.debater_a.split(":", 1)
    b_provider, b_model = args.debater_b.split(":", 1)
    j_provider, j_model = args.judge.split(":", 1)

    env = DebateEnv(
        debater_a_client=LLMClient(provider=a_provider), debater_a_model=a_model,  # type: ignore[arg-type]
        debater_b_client=LLMClient(provider=b_provider), debater_b_model=b_model,  # type: ignore[arg-type]
        judge_client=LLMClient(provider=j_provider), judge_model=j_model,  # type: ignore[arg-type]
        turns_per_debater=args.turns,
    )
    pool = JudgePool([(j_provider, j_model, None) for _ in range(args.judge_pool_size)])
    rubric = SophistryRubric(judge_pool=pool)

    print("\n=== Head-to-head debate ===")
    print(f"Item: article_id={item.article_id}  (passage {len(item.article)} chars)")
    print(f"Q: {task.question}")
    print()
    print(f"Debater A: {args.debater_a}")
    print(f"  argues GOLD answer: {task.debater_a_answer}")
    print(f"Debater B: {args.debater_b}")
    print(f"  argues DISTRACTOR:  {task.debater_b_answer}")
    print()

    print(">>> Rolling out...")
    traj = await env.rollout(task)
    print(f">>> Winner per judge: {traj.ruling.winner}")
    print()

    for i, t in enumerate(traj.turns, start=1):
        print(f"--- Turn {i} ({t.debater}) ---")
        snippet = t.text.strip()
        print(snippet[:600] + ("..." if len(snippet) > 600 else ""))
        print(f"  [{len(t.parsed.claims)} claims, {len(t.parsed.citations)} citations]")
        print()

    print(">>> Scoring with 7-axis rubric (this calls the LLM judges)...")
    scores = await rubric.score(traj)
    print()

    a_label = a_model[:24]
    b_label = b_model[:24]
    print(f"{'axis':<22} {('A: ' + a_label):>28} {('B: ' + b_label):>28}")
    print("-" * 78)
    for axis in (
        "correctness",
        "citation_bluffing",
        "sycophantic",
        "false_confidence",
        "gish_gallop",
        "goalpost",
        "reframing",
    ):
        a_s = scores.get(f"{axis}_A", float("nan"))
        b_s = scores.get(f"{axis}_B", float("nan"))
        marker = ""
        if not (a_s != a_s):  # not NaN
            if abs(a_s - b_s) > 0.05:
                marker = "  ← divergent"
        print(f"{axis:<22} {a_s:>28.3f} {b_s:>28.3f}{marker}")
    print("-" * 78)
    print(f"{'aggregate':<22} {scores.get('aggregate', 0.0):>57.3f}")


if __name__ == "__main__":
    asyncio.run(main())
