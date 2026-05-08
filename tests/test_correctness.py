from sophistry_bench.rubric.correctness import score_correctness
from sophistry_bench.environment import Trajectory, JudgeRuling
from sophistry_bench.dataset import DebateTask


def _make_task(gold="noon", a_ans="noon", b_ans="morning") -> DebateTask:
    return DebateTask(
        article_id="x",
        article="text",
        question="q",
        options=["morning", "noon", "evening"],
        gold_index=1,
        round=1,
        debater_a_answer=a_ans,
        debater_b_answer=b_ans,
    )


def test_correctness_winner_is_gold():
    task = _make_task()
    traj = Trajectory(task=task, turns=[], ruling=JudgeRuling(winner="A", reasoning=""))
    scores = score_correctness(traj)
    assert scores["A"] == 1.0
    assert scores["B"] == 0.0
    assert scores["mean"] == 1.0


def test_correctness_winner_is_distractor():
    task = _make_task()
    traj = Trajectory(task=task, turns=[], ruling=JudgeRuling(winner="B", reasoning=""))
    scores = score_correctness(traj)
    assert scores["A"] == 0.0
    assert scores["B"] == 0.0
    assert scores["mean"] == 0.0


def test_correctness_b_is_gold_and_wins():
    task = _make_task(a_ans="morning", b_ans="noon")
    traj = Trajectory(task=task, turns=[], ruling=JudgeRuling(winner="B", reasoning=""))
    scores = score_correctness(traj)
    assert scores["A"] == 0.0
    assert scores["B"] == 1.0
    assert scores["mean"] == 1.0
