"""Loading logged classifier outcomes from JSONL and CSV.

Focus areas: field-alias resolution, correctness derivation from label pairs,
and error messages that carry the exact line number of a broken record.
"""

import pytest

from miscal import (
    EmptyDatasetError,
    FieldMap,
    MiscalError,
    RecordError,
    load_file,
    parse_text,
    record_from_dict,
)


def test_jsonl_with_confidence_and_correct_keeps_raw_records():
    dataset = parse_text(
        '\n{"confidence": 0.9, "correct": true, "id": "a"}\n\n{"confidence": 0.4, "correct": false}\n\n'
    )
    # Blank lines are skipped; everything else round-trips.
    assert dataset.confidences == [0.9, 0.4]
    assert dataset.outcomes == [1, 0]
    assert len(dataset) == 2
    assert dataset.records[0].raw["id"] == "a"


def test_field_aliases_are_recognized():
    for alias in ("confidence", "conf", "p", "prob", "probability", "score"):
        assert record_from_dict({alias: 0.7, "correct": True}).confidence == 0.7
    for alias in ("correct", "is_correct", "hit", "success"):
        assert record_from_dict({"confidence": 0.7, alias: False}).correct is False


def test_correctness_derived_from_predicted_expected_pair():
    hit = record_from_dict({"confidence": 0.8, "predicted": "spam", "expected": "spam"})
    miss = record_from_dict({"confidence": 0.8, "predicted": "spam", "expected": "ham"})
    assert hit.correct is True
    assert miss.correct is False
    # "Spam " vs "spam" is a formatting quirk of LLM output, not a miss.
    fuzzy = record_from_dict({"confidence": 0.8, "predicted": " Spam ", "expected": "spam"})
    assert fuzzy.correct is True


def test_explicit_field_map_overrides_aliases_and_reports_missing_fields():
    fields = FieldMap(confidence="certainty", correct="was_right")
    record = record_from_dict({"certainty": "80%", "was_right": "yes", "confidence": 0.1}, fields)
    assert record.confidence == pytest.approx(0.8)
    assert record.correct is True
    with pytest.raises(RecordError, match="'certainty' not present"):
        record_from_dict({"confidence": 0.5, "correct": True}, FieldMap(confidence="certainty"))


def test_string_and_integer_correctness_forms():
    for value, expected in (("true", True), ("Yes", True), ("1", True), (1, True),
                            ("false", False), ("n", False), ("0", False), (0, False)):
        assert record_from_dict({"confidence": 0.5, "correct": value}).correct is expected, value


def test_uninterpretable_correctness_raises_with_line_number():
    with pytest.raises(RecordError, match=r"line 3: cannot interpret correctness"):
        parse_text(
            '{"confidence": 0.9, "correct": true}\n'
            '{"confidence": 0.8, "correct": false}\n'
            '{"confidence": 0.7, "correct": "sort of"}\n'
        )


def test_malformed_jsonl_reports_the_offending_line():
    with pytest.raises(RecordError, match="line 2: invalid JSON"):
        parse_text('{"confidence": 0.9, "correct": true}\n{not json}\n')
    with pytest.raises(RecordError, match="must be an object"):
        parse_text("[1, 2, 3]\n", fmt="jsonl")


def test_records_missing_required_fields_raise_helpfully():
    with pytest.raises(RecordError, match="no confidence field found"):
        parse_text('{"correct": true}\n')
    with pytest.raises(RecordError, match="predicted/expected pair"):
        parse_text('{"confidence": 0.9, "predicted": "spam"}\n')


def test_csv_with_header_row_skipping_blank_rows(tmp_path):
    path = tmp_path / "run.csv"
    path.write_text("confidence,correct\n0.9,true\n,\n0.4,false\n", encoding="utf-8")
    dataset = load_file(str(path))
    assert dataset.confidences == [0.9, 0.4]
    assert dataset.outcomes == [1, 0]


def test_csv_error_line_numbers_account_for_the_header(tmp_path):
    path = tmp_path / "run.csv"
    path.write_text("confidence,correct\n0.9,true\nbogus,true\n", encoding="utf-8")
    with pytest.raises(RecordError, match="line 3"):
        load_file(str(path))


def test_auto_format_sniffs_jsonl_by_leading_brace_else_csv():
    assert len(parse_text('{"confidence": 0.9, "correct": true}\n', fmt="auto")) == 1
    assert parse_text("p,hit\n0.9,1\n", fmt="auto").confidences == [0.9]


def test_empty_input_and_unknown_format_raise():
    with pytest.raises(EmptyDatasetError):
        parse_text("\n\n", fmt="jsonl")
    with pytest.raises(MiscalError, match="unknown format"):
        parse_text("x", fmt="yaml")


def test_verbalized_confidences_parse_inside_records():
    dataset = parse_text(
        '{"confidence": "very likely", "correct": true}\n'
        '{"confidence": "9/10", "correct": false}\n'
        '{"confidence": "85%", "correct": true}\n'
    )
    assert dataset.confidences == [0.90, 0.9, 0.85]
