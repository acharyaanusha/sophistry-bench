"""Build the HuggingFace dataset folder from the bundled curated QuALITY slice.

Reads src/sophistry_bench/data/quality_dev.json (50 items) and emits:
- artifacts/hf_dataset/data/dev.parquet
- artifacts/hf_dataset/README.md  (dataset card with YAML front-matter)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_JSON = REPO_ROOT / "src" / "sophistry_bench" / "data" / "quality_dev.json"
OUT_DIR = REPO_ROOT / "artifacts" / "hf_dataset"
OUT_DATA = OUT_DIR / "data"
OUT_PARQUET = OUT_DATA / "dev.parquet"
OUT_README = OUT_DIR / "README.md"


DATASET_CARD = """---
license: cc-by-4.0
language:
- en
pretty_name: Sophistry-Bench QuALITY Dev (50-item curated slice)
size_categories:
- n<1K
task_categories:
- multiple-choice
- question-answering
tags:
- debate
- scalable-oversight
- reading-comprehension
- multi-agent
- alignment
configs:
- config_name: default
  data_files:
  - split: dev
    path: data/dev.parquet
dataset_info:
  features:
  - name: article_id
    dtype: string
  - name: article
    dtype: string
  - name: question
    dtype: string
  - name: options
    sequence: string
  - name: gold_label
    dtype: int32
  splits:
  - name: dev
    num_examples: 50
---

# Sophistry-Bench QuALITY Dev Slice

A 50-item curated subset of the [QuALITY](https://nyu-mll.github.io/quality/)
multiple-choice reading-comprehension dev set, used as the evaluation
distribution for [`sophistry-bench`](https://github.com/acharyaanusha/sophistry-bench) —
an asymmetric-information debate RL environment reproducing the protocol from
Khan et al. 2024 (*Debating with More Persuasive LLMs Leads to More Truthful
Answers*).

## What this slice is for

Sophistry-Bench debates run two LLMs (one defending the gold answer, one
defending a distractor) over a passage that the judge cannot see. The 50
items here are the bundled fallback that the env loads when the upstream
`emozilla/quality` repo is unreachable. They span article length, genre, and
question difficulty.

If you want the full QuALITY dev split, fetch
[`emozilla/quality`](https://huggingface.co/datasets/emozilla/quality)
directly — this dataset is the *curated slice*, not a replacement.

## Schema

| field        | type           | description                                                    |
|--------------|----------------|----------------------------------------------------------------|
| `article_id` | string         | QuALITY article identifier                                     |
| `article`    | string         | Full passage text (debaters see it, judge does not)            |
| `question`   | string         | Multiple-choice question stem                                  |
| `options`    | list[string]   | Four answer choices, in original order                         |
| `gold_label` | int (0-3)      | Index into `options` of the correct answer                     |

## Loading

```python
from datasets import load_dataset

ds = load_dataset("anushaacharya/sophistry-bench-quality-dev", split="dev")
print(ds[0]["question"])
```

## License & attribution

This slice is redistributed under **CC-BY-4.0**, matching the upstream
QuALITY license. The articles are drawn from Project Gutenberg, the Open
American National Corpus, and other sources curated by the QuALITY authors.

If you use this slice, please cite QuALITY:

```bibtex
@inproceedings{pang-etal-2022-quality,
    title = "{Q}u{ALITY}: Question Answering with Long Input Texts, Yes!",
    author = "Pang, Richard Yuanzhe and Parrish, Alicia and Joshi, Nitish and Nangia, Nikita and Phang, Jason and Chen, Angelica and Padmakumar, Vishakh and Ma, Johnny and Thompson, Jana and He, He and Bowman, Samuel R.",
    booktitle = "NAACL 2022",
    year = "2022",
    url = "https://arxiv.org/abs/2112.08608",
}
```

And the debate protocol:

```bibtex
@article{khan2024debating,
  title={Debating with More Persuasive {LLM}s Leads to More Truthful Answers},
  author={Khan, Akbir and Hughes, John and Valentine, Dan and Ruis, Laura and Sachan, Kshitij and Radhakrishnan, Ansh and Grefenstette, Edward and Bowman, Samuel R. and Rockt{\\"a}schel, Tim and Perez, Ethan},
  journal={arXiv preprint arXiv:2402.06782},
  year={2024},
}
```
"""


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)

    with SRC_JSON.open() as f:
        items = json.load(f)

    assert isinstance(items, list) and len(items) == 50, (
        f"Expected 50 items in {SRC_JSON}, got {len(items)}"
    )
    for i, item in enumerate(items):
        expected_keys = {"article_id", "article", "question", "options", "gold_label"}
        missing = expected_keys - set(item.keys())
        assert not missing, f"Item {i} missing keys: {missing}"
        assert len(item["options"]) == 4, f"Item {i} has {len(item['options'])} options, expected 4"
        assert 0 <= item["gold_label"] < 4, f"Item {i} has gold_label={item['gold_label']} out of range"

    df = pd.DataFrame(items)
    df["article_id"] = df["article_id"].astype(str)
    df["gold_label"] = df["gold_label"].astype("int32")
    df.to_parquet(OUT_PARQUET, index=False)

    OUT_README.write_text(DATASET_CARD)

    print(f"Wrote {OUT_PARQUET} ({OUT_PARQUET.stat().st_size} bytes)")
    print(f"Wrote {OUT_README}")


if __name__ == "__main__":
    main()
