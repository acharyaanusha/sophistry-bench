import importlib.util
import json
from pathlib import Path


def _load_finetune_module():
    spec = importlib.util.spec_from_file_location(
        "finetune", Path(__file__).parent.parent / "scripts" / "finetune.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_to_openai_dpo_format_matches_api_schema():
    finetune = _load_finetune_module()
    trl = {"prompt": "Argue X", "chosen": "X is right", "rejected": "Y is right"}
    out = finetune._to_openai_dpo_format(trl)
    assert out["input"]["messages"] == [{"role": "user", "content": "Argue X"}]
    assert out["preferred_output"] == [{"role": "assistant", "content": "X is right"}]
    assert out["non_preferred_output"] == [{"role": "assistant", "content": "Y is right"}]
