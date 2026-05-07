import json
from pathlib import Path

import pytest

from sophistry_bench.dataset import QualityItem, load_quality_from_json


def test_load_quality_returns_typed_items():
    fixture = Path(__file__).parent / "fixtures" / "quality_sample.json"
    items = load_quality_from_json(fixture)
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, QualityItem)
    assert item.question == "When did the expedition reach the summit?"
    assert item.gold_index == 1
    assert item.gold_answer == "Tuesday morning"
    assert len(item.distractors) == 3
    assert "Monday morning" in item.distractors


def test_quality_item_validates_gold_index():
    with pytest.raises(ValueError, match="gold_index"):
        QualityItem(
            article_id="x",
            article="text",
            question="q",
            options=["a", "b"],
            gold_index=5,
        )
