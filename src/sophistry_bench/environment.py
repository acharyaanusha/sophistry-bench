from dataclasses import dataclass, field
from typing import Literal

from sophistry_bench.dataset import DebateTask
from sophistry_bench.parser import ParsedTurn


@dataclass
class DebaterTurn:
    debater: Literal["A", "B"]
    text: str
    parsed: ParsedTurn


@dataclass
class JudgeRuling:
    winner: Literal["A", "B"]
    reasoning: str


@dataclass
class Trajectory:
    task: DebateTask
    turns: list[DebaterTurn] = field(default_factory=list)
    ruling: JudgeRuling | None = None
