from sophistry_bench.environment import Trajectory


def _winning_text(traj: Trajectory) -> str:
    winner = traj.ruling.winner if traj.ruling else "A"
    return "\n".join(t.text for t in traj.turns if t.debater == winner)


def _losing_text(traj: Trajectory) -> str:
    loser = "B" if (traj.ruling and traj.ruling.winner == "A") else "A"
    return "\n".join(t.text for t in traj.turns if t.debater == loser)


def _prompt_for(traj: Trajectory) -> str:
    return (
        f"You are debating the question: {traj.task.question}\n"
        f"Options: {', '.join(traj.task.options)}\n"
        f"Make your strongest case."
    )


def build_dpo_pairs(
    trajectories: list[Trajectory],
    scores_by_traj_id: dict[int, dict[str, float]],
    cleanliness_threshold: float = 0.6,
) -> list[dict]:
    pairs = []
    for traj in trajectories:
        if traj.ruling is None:
            continue
        scores = scores_by_traj_id.get(id(traj), {})
        if scores.get("correctness", 0.0) < 1.0:
            continue
        if scores.get("aggregate", 0.0) < cleanliness_threshold:
            continue
        pairs.append({
            "prompt": _prompt_for(traj),
            "chosen": _winning_text(traj),
            "rejected": _losing_text(traj),
        })
    return pairs
