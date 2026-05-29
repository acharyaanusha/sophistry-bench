"""Llama-3.2-1B smoke test for the Reward Hacking Sprint env.

Verifies that a small instruction-tuned model can actually follow the
``<claim>``/``<cite>`` format from the sprint prompt before burning Sprint
queue compute on 100 GRPO steps.

The script samples N rows from the sprint dataset, calls an OpenAI-compatible
endpoint for each, parses the responses, and reports:

- Format compliance: % of responses with ≥1 ``<claim>`` and ≥1 ``<cite>``
- Claim count distribution (relevant to the pre-registered cliff hypothesis)
- Citation grounding rate (reused from the main rubric)
- A few sample completions for qualitative inspection

Endpoint configuration (any OpenAI-compatible API works — Together, Groq,
Ollama, vLLM, HF Inference, etc.):

    # Together AI
    export OPENAI_API_KEY=$TOGETHER_API_KEY
    export OPENAI_BASE_URL=https://api.together.xyz/v1
    .venv/bin/python scripts/sprint_smoke_test.py \\
        --model meta-llama/Llama-3.2-1B-Instruct --n 10

    # Local Ollama
    export OPENAI_API_KEY=ollama
    export OPENAI_BASE_URL=http://localhost:11434/v1
    .venv/bin/python scripts/sprint_smoke_test.py \\
        --model llama3.2:1b --n 10

Exits non-zero if format compliance < 50% — the prompt needs simplification
before submission if Llama-1B can't follow it on most rollouts.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from statistics import mean, median

from openai import AsyncOpenAI

from sophistry_bench.parser import parse_turn
from sophistry_bench.sprint.env import (
    citation_grounding,
    claim_count_cliff,
    load_environment,
)


async def _one_rollout(
    client: AsyncOpenAI,
    model: str,
    row: dict,
    max_tokens: int,
    temperature: float,
) -> dict:
    """Call the model for one dataset row and score the response."""
    resp = await client.chat.completions.create(
        model=model,
        messages=row["prompt"],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = resp.choices[0].message.content or ""
    parsed = parse_turn(text)
    info = row["info"]
    return {
        "text": text,
        "n_claims": len(parsed.claims),
        "n_citations": len(parsed.citations),
        "cliff_reward": claim_count_cliff(len(parsed.claims)),
        "grounding": citation_grounding(parsed.claims, parsed.citations, info["passage"]),
        "is_gold": info["is_gold"],
        "assigned_answer": info["assigned_answer"],
    }


async def main_async(args: argparse.Namespace) -> int:
    env = load_environment(n_items=args.n_items, passage_chars=args.passage_chars)
    rows = list(env.dataset.select(range(min(args.n, len(env.dataset)))))
    print(
        f"Smoke test: model={args.model} endpoint={args.base_url or 'env-default'} "
        f"rollouts={len(rows)} max_tokens={args.max_tokens}"
    )
    print("-" * 70)

    # AsyncOpenAI() honors OPENAI_API_KEY and OPENAI_BASE_URL from env vars,
    # or accepts them as explicit kwargs.
    client_kwargs: dict = {}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    client = AsyncOpenAI(**client_kwargs)

    sem = asyncio.Semaphore(args.concurrency)

    async def bounded(row: dict) -> dict:
        async with sem:
            try:
                return await _one_rollout(
                    client, args.model, row, args.max_tokens, args.temperature
                )
            except Exception as e:
                return {"error": f"{type(e).__name__}: {e}", "n_claims": 0, "n_citations": 0,
                        "cliff_reward": 0.0, "grounding": 0.0, "is_gold": False,
                        "assigned_answer": row["info"]["assigned_answer"], "text": ""}

    results = await asyncio.gather(*(bounded(r) for r in rows))

    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]

    if not ok:
        print(f"ALL {len(results)} ROLLOUTS FAILED.")
        for e in errors[:3]:
            print(f"  - {e['error']}")
        return 2

    has_claims = sum(1 for r in ok if r["n_claims"] >= 1) / len(ok)
    has_cites = sum(1 for r in ok if r["n_citations"] >= 1) / len(ok)
    has_both = sum(1 for r in ok if r["n_claims"] >= 1 and r["n_citations"] >= 1) / len(ok)
    n_claims_vals = [r["n_claims"] for r in ok]
    n_cites_vals = [r["n_citations"] for r in ok]
    cliff_vals = [r["cliff_reward"] for r in ok]
    ground_vals = [r["grounding"] for r in ok]

    print(f"Completed: {len(ok)}/{len(results)} ({len(errors)} errors)")
    print("Format compliance:")
    print(f"  ≥1 <claim>:      {has_claims:.0%}")
    print(f"  ≥1 <cite>:       {has_cites:.0%}")
    print(f"  ≥1 of both:      {has_both:.0%}")
    print(f"n_claims:    mean={mean(n_claims_vals):.2f} median={median(n_claims_vals):.0f} "
          f"min={min(n_claims_vals)} max={max(n_claims_vals)}")
    print(f"n_citations: mean={mean(n_cites_vals):.2f} median={median(n_cites_vals):.0f} "
          f"min={min(n_cites_vals)} max={max(n_cites_vals)}")
    print(f"cliff reward:     mean={mean(cliff_vals):.3f}  (peaks at 1.0 for n_claims=8)")
    print(f"citation grounding: mean={mean(ground_vals):.3f}  (1.0 = all verbatim, 0.0 = all fabricated)")

    print()
    print("Sample completions (first 3):")
    for i, r in enumerate(ok[:3]):
        side = "GOLD" if r["is_gold"] else "DISTRACTOR"
        snippet = r["text"][:300].replace("\n", " ")
        print(f"  [{i+1}] side={side} answer={r['assigned_answer']!r} "
              f"n_claims={r['n_claims']} n_cites={r['n_citations']}")
        print(f"      {snippet}{'...' if len(r['text']) > 300 else ''}")

    if errors:
        print()
        print(f"Errors ({len(errors)}):")
        for e in errors[:5]:
            print(f"  - {e['error']}")

    print("-" * 70)
    if has_both < 0.5:
        print(f"FAIL: format compliance below 50% ({has_both:.0%}). "
              f"Simplify the prompt before submitting to the Sprint.")
        return 1
    print(f"PASS: format compliance {has_both:.0%}. Prompt is viable for sprint training.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct",
                   help="Model id (provider-specific slug).")
    p.add_argument("--base-url", default=None,
                   help="OpenAI-compatible endpoint URL. Falls back to OPENAI_BASE_URL env var.")
    p.add_argument("--api-key", default=None,
                   help="API key. Falls back to OPENAI_API_KEY env var.")
    p.add_argument("--n", type=int, default=10, help="Number of rollouts.")
    p.add_argument("--n-items", type=int, default=10,
                   help="QuALITY items to load (each yields 2 advocacy rows).")
    p.add_argument("--passage-chars", type=int, default=2000,
                   help="Passage truncation cap (matches sprint env default).")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Per-rollout completion cap (matches sprint-config.toml).")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
