import json
from pathlib import Path

from pydantic import BaseModel, ValidationInfo, field_validator


class QualityItem(BaseModel):
    article_id: str
    article: str
    question: str
    options: list[str]
    gold_index: int

    @field_validator("gold_index")
    @classmethod
    def gold_in_range(cls, v: int, info: ValidationInfo) -> int:
        options = info.data.get("options") or []
        if not (0 <= v < len(options)):
            raise ValueError(f"gold_index {v} out of range for {len(options)} options")
        return v

    @property
    def gold_answer(self) -> str:
        return self.options[self.gold_index]

    @property
    def distractors(self) -> list[str]:
        return [opt for i, opt in enumerate(self.options) if i != self.gold_index]


def load_quality_from_json(path: Path) -> list[QualityItem]:
    raw = json.loads(Path(path).read_text())
    return [
        QualityItem(
            article_id=r["article_id"],
            article=r["article"],
            question=r["question"],
            options=r["options"],
            gold_index=r["gold_label"],
        )
        for r in raw
    ]


class DebateTask(BaseModel):
    article_id: str
    article: str
    question: str
    options: list[str]
    gold_index: int
    round: int
    debater_a_answer: str
    debater_b_answer: str

    @property
    def gold_answer(self) -> str:
        return self.options[self.gold_index]


def build_debate_tasks(item: QualityItem, distractor_index: int) -> list[DebateTask]:
    if distractor_index == item.gold_index:
        raise ValueError("distractor_index must differ from gold_index")
    if not (0 <= distractor_index < len(item.options)):
        raise ValueError("distractor_index out of range")

    distractor = item.options[distractor_index]
    base = dict(
        article_id=item.article_id,
        article=item.article,
        question=item.question,
        options=item.options,
        gold_index=item.gold_index,
    )
    return [
        DebateTask(**base, round=1, debater_a_answer=item.gold_answer, debater_b_answer=distractor),
        DebateTask(**base, round=2, debater_a_answer=distractor, debater_b_answer=item.gold_answer),
    ]
