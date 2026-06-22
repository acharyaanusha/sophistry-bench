"""Guards the public API the OpenEnv server depends on."""
import asyncio
import importlib


def test_public_scoring_and_dataset_api_importable():
    m = importlib.import_module("sophistry_bench_sprint")
    expected = {
        "load_environment",
        "QualityItem",
        "parse_claims",
        "parse_citations",
        "claim_count_cliff",
        "citation_grounding",
        "aggregate_reward",
        "alternation_canary",
        "starts_with_canary",
        "length_band_canary",
        "template_echo_canary",
        "pick_distractor",
        "quality_to_advocacy_dataset",
        "load_quality_from_json",
        "packaged_quality_path",
    }
    assert expected.issubset(set(m.__all__))
    for name in expected:
        assert hasattr(m, name), f"missing public symbol: {name}"


def test_public_aliases_are_the_private_impls():
    m = importlib.import_module("sophistry_bench_sprint")
    assert m.parse_claims is m._parse_claims
    assert m.parse_citations is m._parse_citations
    assert m.aggregate_reward is m._aggregate_reward
    assert m.quality_to_advocacy_dataset is m._quality_to_advocacy_dataset
    assert m.pick_distractor is m._pick_distractor
    assert m.load_quality_from_json is m._load_quality_from_json
    assert m.packaged_quality_path is m._packaged_quality_path


def test_public_aggregate_reward_matches_rubric_func():
    """The exported pure aggregate_reward must equal the trained rubric's
    aggregate_reward (index 0) for the same passage — they share one impl."""
    m = importlib.import_module("sophistry_bench_sprint")
    passage = "alpha beta gamma delta epsilon zeta eta theta"
    text = "<claim>alpha</claim><cite>beta gamma delta epsilon zeta</cite>"
    claims = m.parse_claims(text)
    cites = m.parse_citations(text)

    public = m.aggregate_reward(claims, cites, passage)

    rubric = m.load_environment(n_items=2, passage_chars=500, seed=0).rubric
    if not getattr(rubric, "funcs", None) and getattr(rubric, "rubrics", None):
        rubric = rubric.rubrics[0]
    rubric_fn = rubric.funcs[0]
    canonical = asyncio.run(
        rubric_fn(
            prompt=[],
            completion=[{"role": "assistant", "content": text}],
            answer="",
            state={"info": {"passage": passage}},
        )
    )
    assert abs(public - canonical) < 1e-9
