# Sophistry Bench

An RL environment for asymmetric-information debate, with a 6-component sophistry rubric.

## What it is

Two LLMs debate a multiple-choice question about a passage. Both see the passage; the judge does not. One debater argues the correct answer; the other argues a distractor. The judge picks a winner. The verifier scores not just correctness but **how** wins happen: citation bluffing, sycophantic concession, false confidence, gish gallop, goalpost shifting.

Targets publication on the [Prime Intellect Environments Hub](https://www.primeintellect.ai/blog/environments).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
```

## Quick start

```bash
# Smoke-test the environment with a single rollout
python scripts/demo.py
```

## Loading data

QuALITY ships on HuggingFace as `emozilla/quality`. Use either source:

```bash
# Option A — pull directly from HuggingFace and cache locally
python -c "from sophistry_bench.dataset import load_quality_from_hub; \
  load_quality_from_hub(split='validation', limit=200, cache_path='data/quality_dev.json')"

# Option B — bring your own JSON in the QualityItem schema (see fixture file)
```

## Running the leaderboard

```bash
python scripts/run_eval.py \
  --quality-json data/quality_dev.json \
  --debaters openai:gpt-4o anthropic:claude-haiku-4-5 google:gemini-2.5-flash \
  --judge openai:gpt-4o-mini \
  --n-tasks 50 \
  --output leaderboard.json
```

## Generating DPO pairs and fine-tuning

The pair generator runs `--n-samples-per-task` rollouts per (task, side) and
pairs the cleanest argument vs the dirtiest argument *for the same assigned
answer*. This isolates the sophistry signal from the answer-correctness signal.

```bash
python scripts/generate_dpo_pairs.py \
  --quality-json data/quality_train.json \
  --debater openai:gpt-4o-mini \
  --n-items 500 \
  --n-samples-per-task 3 \
  --output dpo_pairs.jsonl

python scripts/finetune.py \
  --pairs-jsonl dpo_pairs.jsonl \
  --provider openai \
  --model gpt-4o-mini-2024-07-18

# After fine-tune completes, re-run the leaderboard with the new model
python scripts/run_eval.py --debaters openai:ft:... ...
python scripts/compare.py --before leaderboard.json --after leaderboard_ft.json
```

## Multi-judge voting

The `JudgePool` defaults to a 3-judge pool with non-zero temperature so that
median voting actually reduces variance. Real bias reduction requires a
*diverse* pool — pass entries with different providers/models if you can.

## Architecture

- `agents.py` — multi-provider LLM client
- `dataset.py` — QuALITY → debate task transform
- `parser.py` — extract claims/citations from generations
- `rubric/` — 6 sub-rubrics (correctness, citation_bluffing, sycophantic, false_confidence, gish_gallop, goalpost) + aggregator
- `environment.py` — multi-turn debate rollout + verifiers `load_environment` factory
- `eval.py` — single-model and cross-model leaderboard
- `train.py` — DPO pair builder

## Tests

```bash
pytest                         # unit tests with mocked LLMs
RUN_INTEGRATION=1 pytest       # also run integration tests against real APIs
```

## Hub publishing

Currently published to: (none — public release pending)

After approval:
```bash
prime env install --path .
prime env push --path .
```
