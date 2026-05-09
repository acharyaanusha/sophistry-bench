# Khan et al. 2024 — Faithfulness Audit

Source: [Khan et al. 2024 — "Debating with More Persuasive LLMs Leads to More Truthful Answers"](https://arxiv.org/abs/2402.06782) (PDF copy at `docs/references/khan-2024.pdf`).

> **Note on paper identity:** The task brief cited arxiv ID 2407.04622, which is actually Kenton et al. 2024 ("On scalable oversight with weak LLMs judging strong LLMs" — a different paper by a different group). The canonical Khan et al. 2024 debate paper is **arxiv 2402.06782**. The PDF at `docs/references/khan-2024.pdf` has been updated to contain 2402.06782. All values below are drawn from that paper.

This audit maps Khan et al.'s debate protocol to our implementation. Used by Task 1.5.2 to fix unintentional divergences.

| Parameter | Khan et al. value | Sophistry Bench value | Status |
|---|---|---|---|
| Debate length (turns/debater) | 3 rounds; each round both debaters produce one argument simultaneously, so 3 turns per debater | 3 (default in `environment.py:90`) | match |
| Judge model relative to debaters | Non-expert by information asymmetry only (judge lacks passage access); Khan et al. test GPT-4-Turbo, GPT-3.5-Turbo, and Claude 2.1 as judges — same capability tier as or weaker than debaters (GPT-4-Turbo BoN=16) | Defaults pair stronger debater with weaker judge (`vf_env.py:302-303`: `claude-sonnet-4-6` debater + `claude-haiku-4-5` judge; `demo.py` and `run_eval.py` follow the same pattern) | match |
| Debater models | Main results: GPT-4-Turbo (best-of-16) on both sides; cross-play also uses GPT-3.5-Turbo, Claude 2.1, Claude 1.3 — same model on both sides | Same model both sides; configurable via CLI | match |
| Round structure (side swap) | Each question debated twice: once with D1 arguing correct / D2 arguing distractor, then flipped; results averaged over both orderings | 2 rounds, sides swapped (`dataset.py:133-161`) — `build_debate_tasks` returns both orderings | match |
| Distractor selection | "Best distractor": the most plausible wrong-answer option from the QuALITY dataset as pre-labeled — NOT random selection from remaining options | Seeded uniform-random pick from non-gold options (`dataset.py:38-46` `pick_distractor`); deterministic per `(item, seed)` | divergence (documented) |
| Judge sees passage | No — judges "are not allowed access to the original comprehension text" | No — judge prompt (`environment.py:48-59`) contains no passage field | match |
| Debater opening turn symmetry | Simultaneous: "In each round, debaters see the arguments from previous rounds and simultaneously generate their arguments" — both debaters open independently in round 1 with no cross-visibility | Simultaneous: `environment.py:102-108` runs both debaters via `asyncio.gather` with the same `prior_transcript` (None in round 1); neither sees the other's current-round argument | match |
| QuALITY split | T_L: 400 train-set questions (hard subset, Gutenberg sci-fi, untimed annotators chose correct, rated unambiguous) | `"train"` split via `load_quality_from_hub` (`dataset.py:63`); auto-fetched on first `load_environment()` if no JSON is present, capped at 400 items by default. Hard-subset / sci-fi-only filtering NOT applied. | fix |
| Reported judge accuracy (headline) | 76% LLM-judge accuracy in debate vs 48% naive baseline (no debate); debater = GPT-4-Turbo BoN=16, judges = GPT-4-Turbo / GPT-3.5-Turbo / Claude 2.1, 3 rounds, T_L split | TBD (Phase 4.5 target) | reproduction target |

---

## Match (no action needed)

- **Debate length (turns/debater):** Our default of 3 turns per debater matches Khan et al.'s 3-round protocol.
- **Debater models (same-model-both-sides):** Khan et al. use the same model on both sides; we do too, configurable via CLI.
- **Round structure (side swap):** `build_debate_tasks` generates both orderings (round=1 and round=2 with sides flipped), matching the paper's "flip assignments" methodology.
- **Judge sees passage:** Both implementations withhold the passage from the judge.
- **Simultaneous turns:** Both A and B receive the same prior-round transcript and generate via `asyncio.gather`, so neither sees the other's current-round argument before writing — Khan et al.'s "simultaneous" protocol.
- **Capability gap:** Defaults pair a stronger debater with a weaker judge (`claude-sonnet-4-6` debater + `claude-haiku-4-5` judge), matching Khan et al.'s strong-debater / weaker-judge configuration. Self-play is still configurable.

---

## Divergence — documented and acceptable

- **Distractor selection:** Khan et al. use the "best distractor" — the most plausible wrong answer as pre-labeled by timed human annotators in the QuALITY dataset. We use seeded uniform-random selection from non-gold options (`pick_distractor` in `dataset.py:38-46`). This makes debates somewhat easier on average than Khan et al.'s setup — random distractors include easy ones, whereas the "best distractor" is the hardest. Rationale: the `emozilla/quality` HuggingFace mirror does not expose annotator-level data needed to recover the "best distractor" label. Fixing this is gated on a data-engineering task (pull the original NYU MLL release with full annotations).

---

## Divergence — open

1. **QuALITY split / hard-subset filter:** Khan et al. use the T_L split — 400 train-set questions filtered to (a) untimed annotators all chose correctly, (b) annotators marked the question unambiguous, (c) Gutenberg sci-fi stories only. We load the train split via `load_quality_from_hub` and cap at 400 items, but do NOT apply the three filters. The unfiltered train split is easier than T_L on average. Reproducing T_L requires the full annotator-level metadata which the HuggingFace mirror does not expose. Closing this divergence is gated on the same data-engineering task as the best-distractor label.

2. **Debater Best-of-N sampling:** Khan et al.'s headline result uses GPT-4-Turbo with Best-of-16 sampling on the debater (judge picks the best of 16 candidate arguments per turn). We sample once per turn. BoN sampling raises debater quality and is a major contributor to the 76% headline number; without it, expect lower judge accuracy. Adding BoN is straightforward (sample N, score with a critic, pick best) but multiplies inference cost N×.

---

## Notes for an optional headline reproduction (future)

- **Headline reproduction target:** Khan et al. report **76% LLM-judge accuracy** in debate vs **48% naive baseline**, using debater = GPT-4-Turbo (best-of-16 sampling), judge = GPT-4-Turbo / GPT-3.5-Turbo / Claude 2.1 (averaged across judges), 3 rounds simultaneous, T_L split (400 train-set questions), QuALITY hard subset.
- **Model substitution:** Reasonable modern substitutions: GPT-4-Turbo → GPT-4o or claude-opus-4 (debater); GPT-3.5-Turbo → GPT-4o-mini or claude-haiku-4-5 (judge). The comparison stays meaningful as long as the capability gap is preserved.
- **Prerequisites:** the T_L hard-subset filter, best-distractor labels, and Best-of-N debater sampling are all gated on data-engineering work; they're the open divergences listed above. Headline accuracy is not Sophistry Bench's shipping claim — this is an RL environment, not a benchmark reproduction.
- **Kenton et al. 2024 (2407.04622):** Different paper (Google DeepMind, scalable oversight with weak judges). Not Khan et al. — see top-of-file note.
