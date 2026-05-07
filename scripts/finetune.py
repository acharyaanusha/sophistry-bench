import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from openai import AsyncOpenAI


def _to_openai_dpo_format(trl_pair: dict) -> dict:
    return {
        "input": {"messages": [{"role": "user", "content": trl_pair["prompt"]}]},
        "preferred_output": [{"role": "assistant", "content": trl_pair["chosen"]}],
        "non_preferred_output": [{"role": "assistant", "content": trl_pair["rejected"]}],
    }


async def _openai_dpo(pairs_jsonl: Path, model: str, suffix: str) -> str:
    client = AsyncOpenAI()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for line in pairs_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            tmp.write(json.dumps(_to_openai_dpo_format(json.loads(line))) + "\n")
        converted_path = Path(tmp.name)
    try:
        with converted_path.open("rb") as f:
            upload = await client.files.create(file=f, purpose="fine-tune")
        job = await client.fine_tuning.jobs.create(
            training_file=upload.id,
            model=model,
            method={"type": "dpo"},
            suffix=suffix,
        )
        return job.id
    finally:
        converted_path.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-jsonl", type=Path, required=True)
    p.add_argument("--provider", default="openai")
    p.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--suffix", default="sophistry-bench")
    args = p.parse_args()
    if args.provider == "openai":
        job_id = asyncio.run(_openai_dpo(args.pairs_jsonl, args.model, args.suffix))
        print(f"Started OpenAI DPO job: {job_id}")
    else:
        raise SystemExit(f"Provider not yet supported: {args.provider}")


if __name__ == "__main__":
    main()
