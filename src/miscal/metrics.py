"""Calibration metrics computed with the standard library only.

All functions take parallel lists: ``confidences`` (probabilities in [0, 1])
and ``outcomes`` (1 = the classifier was right, 0 = it was wrong). The
binned metrics (ECE, MCE, decomposition) accept precomputed bins so the CLI
computes one binning and reuses it for the report, the gate, and the SVG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .binning import Bin, make_bins
from .errors import EmptyDatasetError

_EPS = 1e-15


def _require_nonempty(confidences: list[float]) -> None:
    if not confidences:
        raise EmptyDatasetError("metric requires at least one record")


def accuracy(outcomes: list[int]) -> float:
    """Fraction of decisions that were correct."""
    if not outcomes:
        raise EmptyDatasetError("metric requires at least one record")
    return sum(outcomes) / len(outcomes)


def mean_confidence(confidences: list[float]) -> float:
    """Average stated confidence across all records."""
    _require_nonempty(confidences)
    return sum(confidences) / len(confidences)


def confidence_gap(confidences: list[float], outcomes: list[int]) -> float:
    """Signed global gap: ``mean confidence - accuracy``.

    Positive = overconfident, negative = underconfident. This is the single
    number that answers "does the model trust itself too much overall?" —
    but it can hide compensating errors, which is why ECE exists.
    """
    return mean_confidence(confidences) - accuracy(outcomes)


def ece(bins: list[Bin]) -> float:
    """Expected Calibration Error: count-weighted mean absolute bin gap."""
    total = sum(b.count for b in bins)
    if total == 0:
        raise EmptyDatasetError("ECE requires at least one non-empty bin")
    return sum(b.count * abs(b.gap) for b in bins) / total


def mce(bins: list[Bin]) -> float:
    """Maximum Calibration Error: the worst absolute gap in any occupied bin."""
    occupied = [abs(b.gap) for b in bins if b.count > 0]
    if not occupied:
        raise EmptyDatasetError("MCE requires at least one non-empty bin")
    return max(occupied)


def expected_calibration_error(
    confidences: list[float], outcomes: list[int], n_bins: int = 10, strategy: str = "width"
) -> float:
    """Convenience wrapper: bin then compute ECE in one call."""
    return ece(make_bins(confidences, outcomes, n_bins=n_bins, strategy=strategy))


def brier_score(confidences: list[float], outcomes: list[int]) -> float:
    """Mean squared error between stated confidence and the 0/1 outcome."""
    _require_nonempty(confidences)
    return sum((c - y) ** 2 for c, y in zip(confidences, outcomes)) / len(confidences)


def brier_decomposition(outcomes: list[int], bins: list[Bin]) -> tuple[float, float, float]:
    """Murphy decomposition of the Brier score: (reliability, resolution, uncertainty).

    * reliability — how far bin confidence sits from bin accuracy (lower is better);
    * resolution  — how far bin accuracies spread from the base rate (higher is better);
    * uncertainty — base-rate variance, a property of the task, not the model.

    ``reliability - resolution + uncertainty`` equals the Brier score exactly
    when confidences are constant within each bin, and differs only by the
    within-bin confidence variance otherwise.
    """
    total = sum(b.count for b in bins)
    if total == 0 or not outcomes:
        raise EmptyDatasetError("Brier decomposition requires at least one record")
    base_rate = sum(outcomes) / len(outcomes)
    reliability = sum(b.count * (b.mean_confidence - b.accuracy) ** 2 for b in bins) / total
    resolution = sum(b.count * (b.accuracy - base_rate) ** 2 for b in bins) / total
    uncertainty = base_rate * (1.0 - base_rate)
    return reliability, resolution, uncertainty


def log_loss(confidences: list[float], outcomes: list[int]) -> float:
    """Negative log-likelihood of the outcomes under the stated confidences.

    Confidences are clamped away from exact 0/1 so a single overconfident
    miss produces a large-but-finite penalty instead of infinity.
    """
    _require_nonempty(confidences)
    total = 0.0
    for conf, outcome in zip(confidences, outcomes):
        p = min(max(conf, _EPS), 1.0 - _EPS)
        total += -math.log(p) if outcome else -math.log(1.0 - p)
    return total / len(confidences)


@dataclass(frozen=True)
class Metrics:
    """Every scalar miscal reports for one dataset, plus the bins used."""

    n_records: int
    accuracy: float
    mean_confidence: float
    confidence_gap: float
    ece: float
    adaptive_ece: float
    mce: float
    brier: float
    brier_reliability: float
    brier_resolution: float
    brier_uncertainty: float
    log_loss: float
    n_bins: int
    strategy: str
    bins: tuple[Bin, ...]

    @property
    def verdict(self) -> str:
        """Plain-English calibration verdict based on the signed global gap."""
        if self.confidence_gap >= 0.02:
            return "overconfident"
        if self.confidence_gap <= -0.02:
            return "underconfident"
        return "well-calibrated"


def evaluate(
    confidences: list[float], outcomes: list[int], n_bins: int = 10, strategy: str = "width"
) -> Metrics:
    """Compute the full metric set with a single pass of binning."""
    bins = make_bins(confidences, outcomes, n_bins=n_bins, strategy=strategy)
    mass_bins = (
        bins if strategy == "mass" else make_bins(confidences, outcomes, n_bins, "mass")
    )
    reliability, resolution, uncertainty = brier_decomposition(outcomes, bins)
    return Metrics(
        n_records=len(confidences),
        accuracy=accuracy(outcomes),
        mean_confidence=mean_confidence(confidences),
        confidence_gap=confidence_gap(confidences, outcomes),
        ece=ece(bins),
        adaptive_ece=ece(mass_bins),
        mce=mce(bins),
        brier=brier_score(confidences, outcomes),
        brier_reliability=reliability,
        brier_resolution=resolution,
        brier_uncertainty=uncertainty,
        log_loss=log_loss(confidences, outcomes),
        n_bins=n_bins,
        strategy=strategy,
        bins=tuple(bins),
    )
