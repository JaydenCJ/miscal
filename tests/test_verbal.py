"""Parsing every supported confidence representation into a probability.

The scale rules here are load-bearing: a silent misparse would corrupt every
downstream metric, so the failure cases matter as much as the happy paths.
"""

import math

import pytest

from miscal import ConfidenceError, parse_confidence
from miscal.verbal import WORD_SCALE


def test_numbers_in_the_unit_interval_pass_through():
    assert parse_confidence(0.85) == 0.85
    assert parse_confidence(0) == 0.0
    # 1 is ambiguous between "100%" and "p=1.0"; the documented rule says
    # values in [0, 1] are probabilities, so 1 must mean certainty.
    assert parse_confidence(1) == 1.0


def test_numbers_between_one_and_hundred_are_treated_as_percent():
    assert parse_confidence(85) == pytest.approx(0.85)
    assert parse_confidence(99.5) == pytest.approx(0.995)


def test_out_of_range_numbers_are_rejected_not_clamped():
    # Clamping would hide a logging bug; both directions must raise.
    with pytest.raises(ConfidenceError, match="refusing to guess"):
        parse_confidence(-0.2)
    with pytest.raises(ConfidenceError):
        parse_confidence(150)


def test_percent_strings_parse_and_validate():
    assert parse_confidence("85%") == pytest.approx(0.85)
    assert parse_confidence("85 %") == pytest.approx(0.85)
    assert parse_confidence(" 7.5% ") == pytest.approx(0.075)
    with pytest.raises(ConfidenceError, match="outside"):
        parse_confidence("120%")


def test_fraction_strings_parse_and_validate():
    assert parse_confidence("9/10") == pytest.approx(0.9)
    assert parse_confidence("3 / 4") == pytest.approx(0.75)
    with pytest.raises(ConfidenceError, match="zero denominator"):
        parse_confidence("3/0")
    with pytest.raises(ConfidenceError):
        parse_confidence("11/10")


def test_numeric_strings_follow_the_same_scale_rule_as_numbers():
    assert parse_confidence("0.5") == 0.5
    assert parse_confidence("85") == pytest.approx(0.85)


def test_anchor_words_are_case_insensitive_and_tolerate_punctuation():
    assert parse_confidence("very likely") == 0.90
    assert parse_confidence("Very Likely") == 0.90
    assert parse_confidence("Almost certain.") == 0.97


def test_multiword_anchors_do_not_collapse_to_their_last_word():
    # "very unlikely" must not be parsed as "unlikely" (0.08 vs 0.25).
    assert parse_confidence("very unlikely") == 0.08
    assert parse_confidence("unlikely") == 0.25


def test_every_word_scale_anchor_is_a_valid_probability():
    for word, value in WORD_SCALE.items():
        assert 0.0 <= value <= 1.0, word
        assert parse_confidence(word) == value


def test_unknown_word_error_mentions_the_documented_anchor_list():
    with pytest.raises(ConfidenceError, match="anchor words"):
        parse_confidence("meh")


def test_boolean_nan_none_and_empty_string_are_rejected():
    with pytest.raises(ConfidenceError, match="boolean"):
        parse_confidence(True)
    with pytest.raises(ConfidenceError, match="NaN"):
        parse_confidence(math.nan)
    with pytest.raises(ConfidenceError):
        parse_confidence(None)
    with pytest.raises(ConfidenceError, match="empty"):
        parse_confidence("   ")
