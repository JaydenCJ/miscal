"""Calibration metrics against hand-computed reference values.

The reference dataset and its expected numbers are worked out by hand in
``conftest.hand_dataset`` — if an implementation change moves any of these
values, that is a correctness bug, not a tolerance issue.
"""

import math

import pytest

from miscal import (
    EmptyDatasetError,
    accuracy,
    brier_decomposition,
    brier_score,
    confidence_gap,
    ece,
    evaluate,
    expected_calibration_error,
    log_loss,
    make_bins,
    mce,
    mean_confidence,
)
from conftest import hand_dataset


def test_accuracy_mean_confidence_and_gap():
    confs, outs = hand_dataset()
    assert accuracy(outs) == pytest.approx(3 / 7)
    assert mean_confidence(confs) == pytest.approx((0.9 * 4 + 0.6 * 2 + 0.3) / 7)
    gap = confidence_gap(confs, outs)
    assert gap == pytest.approx(mean_confidence(confs) - 3 / 7)
    assert gap > 0  # this dataset is overconfident by construction


def test_ece_matches_the_hand_computation():
    confs, outs = hand_dataset()
    assert expected_calibration_error(confs, outs, n_bins=10) == pytest.approx(0.3)
    # The convenience wrapper agrees with binning manually first.
    bins = make_bins(confs, outs, 10, "width")
    assert expected_calibration_error(confs, outs, 10, "width") == pytest.approx(ece(bins))


def test_mce_is_the_worst_bin_gap():
    confs, outs = hand_dataset()
    bins = make_bins(confs, outs, 10, "width")
    assert mce(bins) == pytest.approx(0.4)


def test_ece_spans_its_full_range_on_degenerate_data():
    # Fully calibrated: confidence 1.0 and always right -> ECE 0.
    assert expected_calibration_error([1.0, 1.0], [1, 1]) == pytest.approx(0.0)
    # Maximally miscalibrated: confidence 1.0 and always wrong -> ECE 1.
    assert expected_calibration_error([1.0, 1.0], [0, 0]) == pytest.approx(1.0)


def test_brier_score_matches_the_hand_computation_and_bounds():
    confs, outs = hand_dataset()
    assert brier_score(confs, outs) == pytest.approx(2.25 / 7)
    assert brier_score([1.0], [1]) == 0.0
    assert brier_score([1.0], [0]) == 1.0


def test_brier_decomposition_components_match_hand_values():
    confs, outs = hand_dataset()
    bins = make_bins(confs, outs, 10, "width")
    reliability, resolution, uncertainty = brier_decomposition(outs, bins)
    assert reliability == pytest.approx(0.75 / 7)
    assert uncertainty == pytest.approx((3 / 7) * (4 / 7))
    assert resolution == pytest.approx((6 * (0.5 - 3 / 7) ** 2 + (3 / 7) ** 2) / 7)


def test_brier_decomposition_identity_holds_for_constant_confidence_bins():
    # Confidences here are constant within each width bin, so the Murphy
    # identity reliability - resolution + uncertainty == Brier is exact.
    confs, outs = hand_dataset()
    bins = make_bins(confs, outs, 10, "width")
    reliability, resolution, uncertainty = brier_decomposition(outs, bins)
    assert reliability - resolution + uncertainty == pytest.approx(brier_score(confs, outs))


def test_log_loss_matches_a_direct_computation_and_stays_finite():
    confs, outs = [0.8, 0.4], [1, 0]
    expected = (-math.log(0.8) - math.log(0.6)) / 2
    assert log_loss(confs, outs) == pytest.approx(expected)
    # A confident miss must be a huge penalty, not an infinity/crash...
    value = log_loss([1.0], [0])
    assert 30 < value < float("inf")
    # ...and perfect predictions cost (numerically) nothing.
    assert log_loss([1.0, 1.0], [1, 1]) == pytest.approx(0.0, abs=1e-12)


def test_worse_calibration_yields_strictly_higher_ece():
    outs = [1, 0] * 20  # 50% accuracy
    mild = [0.6] * 40
    severe = [0.95] * 40
    assert expected_calibration_error(severe, outs) > expected_calibration_error(mild, outs)


def test_evaluate_bundles_consistent_fields():
    confs, outs = hand_dataset()
    metrics = evaluate(confs, outs, n_bins=10, strategy="width")
    assert metrics.n_records == 7
    assert metrics.ece == pytest.approx(0.3)
    assert metrics.brier == pytest.approx(2.25 / 7)
    assert metrics.n_bins == 10
    assert metrics.strategy == "width"
    assert len(metrics.bins) == 10
    assert sum(b.count for b in metrics.bins) == 7


def test_evaluate_adaptive_ece_uses_equal_mass_bins():
    confs, outs = hand_dataset()
    metrics = evaluate(confs, outs, n_bins=3, strategy="width")
    mass_bins = make_bins(confs, outs, 3, "mass")
    assert metrics.adaptive_ece == pytest.approx(ece(mass_bins))


def test_verdict_covers_all_three_regimes():
    assert evaluate([0.7] * 10, [1] * 7 + [0] * 3).verdict == "well-calibrated"
    assert evaluate([0.9] * 10, [1] * 7 + [0] * 3).verdict == "overconfident"
    assert evaluate([0.5] * 10, [1] * 7 + [0] * 3).verdict == "underconfident"


def test_metrics_raise_on_empty_input():
    with pytest.raises(EmptyDatasetError):
        accuracy([])
    with pytest.raises(EmptyDatasetError):
        mean_confidence([])
    with pytest.raises(EmptyDatasetError):
        brier_score([], [])
    with pytest.raises(EmptyDatasetError):
        log_loss([], [])
    with pytest.raises(EmptyDatasetError):
        evaluate([], [])
