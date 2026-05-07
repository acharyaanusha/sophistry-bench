from pathlib import Path

import pytest

from sophistry_bench.dataset import QualityItem, load_quality_from_json, DebateTask, build_debate_tasks


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


def test_build_debate_tasks_yields_two_rounds():
    item = QualityItem(
        article_id="x",
        article="The party arrived at noon.",
        question="When did the party arrive?",
        options=["morning", "noon", "evening", "midnight"],
        gold_index=1,
    )
    tasks = build_debate_tasks(item, distractor_index=0)
    assert len(tasks) == 2
    r1, r2 = tasks
    assert r1.round == 1
    assert r1.debater_a_answer == "noon"
    assert r1.debater_b_answer == "morning"
    assert r1.gold_answer == "noon"
    assert r2.round == 2
    assert r2.debater_a_answer == "morning"
    assert r2.debater_b_answer == "noon"


def test_build_debate_tasks_rejects_invalid_distractor():
    item = QualityItem(
        article_id="x",
        article="t",
        question="q",
        options=["a", "b"],
        gold_index=0,
    )
    with pytest.raises(ValueError, match="distractor"):
        build_debate_tasks(item, distractor_index=0)
