from sophistry_bench.train import build_dpo_pairs
from sophistry_bench.environment import Trajectory, DebaterTurn, JudgeRuling
from sophistry_bench.parser import ParsedTurn
from sophistry_bench.dataset import DebateTask


def _traj(article_id: str, a_text: str, b_text: str, winner: str = "A") -> Trajectory:
    task = DebateTask(
        article_id=article_id, article="The summit was Tuesday.", question="When?",
        options=["Mon","Tue","Wed"], gold_index=1, round=1,
        debater_a_answer="Tue", debater_b_answer="Wed",
    )
    turns = [
        DebaterTurn(debater="A", text=a_text, parsed=ParsedTurn(claims=[],citations=[],raw=a_text)),
        DebaterTurn(debater="B", text=b_text, parsed=ParsedTurn(claims=[],citations=[],raw=b_text)),
    ]
    return Trajectory(task=task, turns=turns, ruling=JudgeRuling(winner=winner, reasoning=""))


def test_pair_built_from_within_side_high_vs_low_aggregate():
    """Two rollouts of the same task; pair gold-side cleanest vs dirtiest."""
    clean = _traj("art1", a_text="Clean A turn", b_text="B turn")
    dirty = _traj("art1", a_text="Dirty A turn", b_text="B turn")
    scored = [
        (clean, {"aggregate": 0.9, "correctness": 1.0}),
        (dirty, {"aggregate": 0.3, "correctness": 1.0}),
    ]
    pairs = build_dpo_pairs(scored, cleanliness_threshold=0.5, min_gap=0.1)
    a_side_pair = next(p for p in pairs if "Clean A turn" in p["chosen"])
    assert "Clean A turn" in a_side_pair["chosen"]
    assert "Dirty A turn" in a_side_pair["rejected"]
    # Prompt now includes the passage and the assigned answer
    assert "PASSAGE:" in a_side_pair["prompt"]
    assert "The summit was Tuesday" in a_side_pair["prompt"]
    assert "ASSIGNED ANSWER: Tue" in a_side_pair["prompt"]


def test_no_pair_when_only_one_rollout_per_side():
    traj = _traj("art1", "A1", "B1")
    pairs = build_dpo_pairs([(traj, {"aggregate": 0.9})], cleanliness_threshold=0.5)
    assert pairs == []


def test_no_pair_when_cleanest_below_threshold():
    t1 = _traj("art1", "A1", "B1")
    t2 = _traj("art1", "A2", "B2")
    scored = [
        (t1, {"aggregate": 0.4}),
        (t2, {"aggregate": 0.2}),
    ]
    pairs = build_dpo_pairs(scored, cleanliness_threshold=0.6)
    assert pairs == []


def test_no_pair_when_gap_too_small():
    t1 = _traj("art1", "A1", "B1")
    t2 = _traj("art1", "A2", "B2")
    scored = [
        (t1, {"aggregate": 0.85}),
        (t2, {"aggregate": 0.84}),
    ]
    pairs = build_dpo_pairs(scored, cleanliness_threshold=0.5, min_gap=0.1)
    assert pairs == []


def test_pairs_grouped_per_side_per_article():
    """One article, two rollouts → 2 pairs (A-side and B-side), each with chosen
    drawn from the higher-aggregate rollout for that side."""
    t1 = _traj("art1", "A_from_t1", "B_from_t1")
    t2 = _traj("art1", "A_from_t2", "B_from_t2")
    scored = [
        (t1, {"aggregate": 0.9}),
        (t2, {"aggregate": 0.3}),
    ]
    pairs = build_dpo_pairs(scored, cleanliness_threshold=0.5, min_gap=0.1)
    assert len(pairs) == 2
    chosens = {p["chosen"] for p in pairs}
    rejecteds = {p["rejected"] for p in pairs}
    assert chosens == {"A_from_t1", "B_from_t1"}
    assert rejecteds == {"A_from_t2", "B_from_t2"}
