"""Temperature scaling: application, fitting, and its promises.

The fit is a deterministic golden-section search, so tests can assert exact
reproducibility and directional guarantees (overconfident data must fit
T > 1, and rescaling must never make the NLL worse than the identity).
"""

import pytest

from miscal import EmptyDatasetError, apply_temperature, fit_temperature, log_loss
from conftest import synthetic_run


def test_temperature_direction_softens_or_sharpens():
    # T = 1 is the identity (up to the epsilon clamp at the extremes).
    confs = [0.1, 0.35, 0.5, 0.72, 0.9]
    assert apply_temperature(confs, 1.0) == pytest.approx(confs, abs=1e-9)
    softened = apply_temperature([0.95, 0.05], 4.0)
    assert 0.5 < softened[0] < 0.95
    assert 0.05 < softened[1] < 0.5
    sharpened = apply_temperature([0.7, 0.3], 0.5)
    assert sharpened[0] > 0.7
    assert sharpened[1] < 0.3


def test_scaling_preserves_ordering_and_the_unit_interval():
    confs = [0.0, 0.2, 0.4, 0.6, 0.8, 0.99, 1.0]
    for temperature in (0.1, 0.3, 2.0, 10.0):
        rescaled = apply_temperature(confs, temperature)
        assert rescaled == sorted(rescaled)
        assert all(0.0 <= p <= 1.0 for p in rescaled)


def test_nonpositive_temperature_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        apply_temperature([0.5], 0.0)
    with pytest.raises(ValueError, match="positive"):
        apply_temperature([0.5], -2.0)


def test_fit_on_calibrated_data_finds_temperature_near_one():
    confs, outs = synthetic_run(n=800, honesty=1.0, seed=11)
    fit = fit_temperature(confs, outs)
    assert 0.8 < fit.temperature < 1.25


def test_fit_on_overconfident_data_finds_temperature_above_one():
    confs, outs = synthetic_run(n=800, honesty=0.3, seed=12)
    fit = fit_temperature(confs, outs)
    assert fit.temperature > 1.5
    assert fit.nll_after < fit.nll_before
    assert "overconfident" in fit.direction
    assert fit.ece_before > 0
    assert fit.nll_before == pytest.approx(log_loss(confs, outs))


def test_fit_on_underconfident_data_finds_temperature_below_one():
    # Flip the direction: outcomes are more often right than stated.
    confs, outs = synthetic_run(n=800, honesty=1.9, seed=13)
    fit = fit_temperature(confs, outs)
    assert fit.temperature < 0.95
    assert "underconfident" in fit.direction


def test_fitted_temperature_never_increases_nll():
    for honesty in (0.3, 1.0, 1.9):
        confs, outs = synthetic_run(n=400, honesty=honesty, seed=14)
        fit = fit_temperature(confs, outs)
        assert fit.nll_after <= fit.nll_before + 1e-9


def test_fit_is_deterministic():
    confs, outs = synthetic_run(n=300, honesty=0.4, seed=15)
    assert fit_temperature(confs, outs) == fit_temperature(confs, outs)


def test_fit_reports_ece_under_the_requested_binning_strategy():
    # The optimal temperature is strategy-independent (it minimizes NLL), but
    # the before/after ECE must honor --strategy, matching `miscal report`.
    confs, outs = synthetic_run(n=400, honesty=0.4, seed=16)
    width = fit_temperature(confs, outs, strategy="width")
    mass = fit_temperature(confs, outs, strategy="mass")
    assert width.temperature == mass.temperature
    assert width.ece_before != mass.ece_before


def test_fit_on_empty_dataset_raises():
    with pytest.raises(EmptyDatasetError):
        fit_temperature([], [])
