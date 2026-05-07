import pytest

from sophistry_bench.train import build_dpo_pairs
from sophistry_bench.environment import Trajectory, DebaterTurn, JudgeRuling
from sophistry_bench.parser import ParsedTurn
from sophistry_bench.dataset import DebateTask


def _traj(round_num: int, winner: str, a_ans: str, b_ans: str, gold: str = "Tue") -> Trajectory:
    task = DebateTask(
        article_id="x", article="The summit was Tuesday.", question="When?",
        options=["Mon","Tue","Wed"], gold_index=1, round=round_num,
        debater_a_answer=a_ans, debater_b_answer=b_ans,
    )
    turns = [
        DebaterTurn(debater="A", text="A turn", parsed=ParsedTurn(claims=[],citations=[],raw="A turn")),
        DebaterTurn(debater="B", text="B turn", parsed=ParsedTurn(claims=[],citations=[],raw="B turn")),
    ]
    return Trajectory(task=task, turns=turns, ruling=JudgeRuling(winner=winner, reasoning=""))


def test_dpo_pair_built_when_correct_side_wins_with_low_sophistry():
    traj = _traj(round_num=1, winner="A", a_ans="Tue", b_ans="Wed")
    scores_by_traj_id = {id(traj): {"aggregate": 0.9, "correctness": 1.0}}
    pairs = build_dpo_pairs([traj], scores_by_traj_id, cleanliness_threshold=0.5)
    assert len(pairs) == 1
    assert "Tue" in pairs[0]["chosen"] or "A turn" in pairs[0]["chosen"]
    assert "B turn" in pairs[0]["rejected"]


def test_dpo_pair_skipped_when_winner_high_sophistry():
    traj = _traj(round_num=1, winner="A", a_ans="Tue", b_ans="Wed")
    scores = {id(traj): {"aggregate": 0.3, "correctness": 1.0}}
    pairs = build_dpo_pairs([traj], scores, cleanliness_threshold=0.5)
    assert pairs == []


def test_dpo_pair_skipped_when_wrong_side_wins():
    traj = _traj(round_num=1, winner="B", a_ans="Tue", b_ans="Wed")
    scores = {id(traj): {"aggregate": 0.9, "correctness": 0.0}}
    pairs = build_dpo_pairs([traj], scores, cleanliness_threshold=0.5)
    assert pairs == []
