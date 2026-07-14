"""Text, JSON, and comparison report rendering."""

import json

import pytest

from miscal import compare_report, evaluate, json_report, text_report
from conftest import hand_dataset


@pytest.fixture()
def metrics():
    confs, outs = hand_dataset()
    return evaluate(confs, outs, n_bins=10, strategy="width")


def test_text_report_lists_every_metric(metrics):
    text = text_report(metrics, source="run.jsonl")
    for label in (
        "accuracy",
        "mean confidence",
        "confidence gap",
        "ECE",
        "adaptive ECE",
        "MCE",
        "Brier score",
        "reliability",
        "resolution",
        "uncertainty",
        "log loss",
    ):
        assert label in text, label
    assert "miscal report — run.jsonl" in text


def test_text_report_has_one_row_per_bin_with_dashes_for_empty_bins(metrics):
    text = text_report(metrics)
    bin_rows = [ln for ln in text.splitlines() if ln.strip().startswith("[")]
    assert len(bin_rows) == metrics.n_bins
    assert "[0.00,0.10]    0       -      -       -" in text


def test_verdict_phrasing_covers_all_three_regimes(metrics):
    assert "verdict: overconfident (stated confidence exceeds accuracy by" in text_report(metrics)
    under = evaluate([0.5] * 10, [1] * 9 + [0])
    assert "verdict: underconfident (accuracy exceeds stated confidence by" in text_report(under)
    calibrated = evaluate([0.7] * 10, [1] * 7 + [0] * 3)
    assert "verdict: well-calibrated (confidence within" in text_report(calibrated)


def test_json_report_is_valid_and_complete(metrics):
    payload = json.loads(json_report(metrics, source="run.jsonl"))
    assert payload["source"] == "run.jsonl"
    assert payload["n_records"] == 7
    assert payload["ece"] == pytest.approx(0.3)
    assert payload["verdict"] == "overconfident"
    assert len(payload["bins"]) == metrics.n_bins
    assert set(payload["bins"][0]) == {"lower", "upper", "count", "mean_confidence", "accuracy", "gap"}


def test_json_report_keys_are_sorted_for_stable_diffs(metrics):
    text = json_report(metrics)
    assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"


def test_compare_report_shows_deltas_and_direction():
    confs, outs = hand_dataset()
    worse = evaluate(confs, outs)
    better = evaluate([0.5] * 10, [1] * 5 + [0] * 5)  # perfectly calibrated
    text = compare_report(worse, better, name_a="old.jsonl", name_b="new.jsonl")
    assert "A: old.jsonl (7 records)" in text
    assert "B: new.jsonl (10 records)" in text
    assert "(B better)" in text
    assert "verdict A: overconfident" in text
    assert "verdict B: well-calibrated" in text
    # And the reverse comparison flags the regression.
    assert "(B worse)" in compare_report(better, worse)
