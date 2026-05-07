from sophistry_bench.parser import ParsedTurn, parse_turn
from tests.fixtures.transcripts import (
    DEBATER_TURN_CLEAN,
    DEBATER_TURN_NO_CITES,
    DEBATER_TURN_MULTIPLE,
    DEBATER_TURN_MALFORMED,
)


def test_parse_clean_turn():
    parsed = parse_turn(DEBATER_TURN_CLEAN)
    assert isinstance(parsed, ParsedTurn)
    assert len(parsed.claims) == 1
    assert parsed.claims[0] == "The expedition reached the summit on Tuesday."
    assert len(parsed.citations) == 1
    assert "Tuesday morning" in parsed.citations[0]


def test_parse_no_cites():
    parsed = parse_turn(DEBATER_TURN_NO_CITES)
    assert len(parsed.claims) == 1
    assert parsed.citations == []


def test_parse_multiple():
    parsed = parse_turn(DEBATER_TURN_MULTIPLE)
    assert len(parsed.claims) == 2
    assert len(parsed.citations) == 3


def test_parse_malformed_returns_empty():
    parsed = parse_turn(DEBATER_TURN_MALFORMED)
    assert parsed.claims == []
    assert parsed.citations == []
