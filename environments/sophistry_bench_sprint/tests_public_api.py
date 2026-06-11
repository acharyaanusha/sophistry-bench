"""Guards the public API the OpenEnv server depends on."""
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
    assert m.quality_to_advocacy_dataset is m._quality_to_advocacy_dataset
    assert m.pick_distractor is m._pick_distractor
    assert m.load_quality_from_json is m._load_quality_from_json
    assert m.packaged_quality_path is m._packaged_quality_path
