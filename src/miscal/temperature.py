"""Temperature scaling: the one-parameter fix for miscalibrated confidences.

Temperature scaling divides every confidence's logit by a constant ``T``
before mapping it back through the sigmoid. ``T > 1`` softens overconfident
scores toward 0.5; ``T < 1`` sharpens underconfident ones. It is the standard
first remedy because it cannot change which answer the model gives — only how
much it claims to trust it.

The fit minimizes negative log-likelihood over ``T`` with a golden-section
search in log-space. NLL as a function of ``log T`` is unimodal for this
one-parameter family, so the search converges deterministically with no
dependence on gradients, learning rates, or random initialization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import EmptyDatasetError
from .metrics import expected_calibration_error, log_loss

# Search range: T below 0.05 or above 20 means the confidences carry almost
# no usable signal; report the boundary instead of chasing infinity.
_LOG_T_MIN = math.log(0.05)
_LOG_T_MAX = math.log(20.0)
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def apply_temperature(confidences: list[float], temperature: float) -> list[float]:
    """Rescale confidences by ``T`` in logit space; ``T=1`` is the identity."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return [_sigmoid(_logit(c) / temperature) for c in confidences]


@dataclass(frozen=True)
class TemperatureFit:
    """Result of fitting a temperature to a dataset."""

    temperature: float
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float

    @property
    def direction(self) -> str:
        """What the fitted temperature says about the raw confidences."""
        if self.temperature > 1.05:
            return "overconfident (confidences softened toward 0.5)"
        if self.temperature < 0.95:
            return "underconfident (confidences sharpened)"
        return "already calibrated (temperature ~ 1)"


def _nll_at(log_t: float, confidences: list[float], outcomes: list[int]) -> float:
    return log_loss(apply_temperature(confidences, math.exp(log_t)), outcomes)


def fit_temperature(
    confidences: list[float], outcomes: list[int], n_bins: int = 10, strategy: str = "width"
) -> TemperatureFit:
    """Find the NLL-minimizing temperature via golden-section search.

    Runs 80 iterations, shrinking the bracket by the golden ratio each step —
    final bracket width is ~1e-16 of the initial range, far below reporting
    precision. Fully deterministic for a given dataset. ``n_bins`` and
    ``strategy`` only affect the reported before/after ECE, not the fit.
    """
    if not confidences:
        raise EmptyDatasetError("temperature fitting requires at least one record")
    lo, hi = _LOG_T_MIN, _LOG_T_MAX
    x1 = hi - _GOLDEN * (hi - lo)
    x2 = lo + _GOLDEN * (hi - lo)
    f1 = _nll_at(x1, confidences, outcomes)
    f2 = _nll_at(x2, confidences, outcomes)
    for _ in range(80):
        if f1 <= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - _GOLDEN * (hi - lo)
            f1 = _nll_at(x1, confidences, outcomes)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + _GOLDEN * (hi - lo)
            f2 = _nll_at(x2, confidences, outcomes)
    temperature = math.exp((lo + hi) / 2.0)
    rescaled = apply_temperature(confidences, temperature)
    return TemperatureFit(
        temperature=temperature,
        nll_before=log_loss(confidences, outcomes),
        nll_after=log_loss(rescaled, outcomes),
        ece_before=expected_calibration_error(confidences, outcomes, n_bins, strategy),
        ece_after=expected_calibration_error(rescaled, outcomes, n_bins, strategy),
    )
