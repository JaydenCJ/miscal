"""Confidence binning strategies for calibration metrics and diagrams.

Two strategies are provided:

* ``width`` — equal-width bins over ``[0, 1]``: the textbook reliability
  diagram, easy to read, but sparse bins are noisy.
* ``mass``  — equal-mass (quantile) bins: every bin holds roughly the same
  number of records, which is what adaptive ECE uses and what you want when
  confidences cluster near 1.0 (as LLM confidences almost always do).

Bins are plain frozen dataclasses so both the metrics layer and the SVG
renderer consume the same structure.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import EmptyDatasetError, MiscalError

STRATEGIES = ("width", "mass")


@dataclass(frozen=True)
class Bin:
    """Aggregated statistics for one confidence bin.

    ``lower``/``upper`` are the bin edges. Empty bins have ``count == 0`` and
    zeroed statistics; they are kept so equal-width diagrams show gaps.
    """

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Signed calibration gap: positive means overconfident in this bin."""
        return self.mean_confidence - self.accuracy


def _validate(confidences: list[float], outcomes: list[int], n_bins: int) -> None:
    if len(confidences) != len(outcomes):
        raise MiscalError(
            f"confidences ({len(confidences)}) and outcomes ({len(outcomes)}) differ in length"
        )
    if not confidences:
        raise EmptyDatasetError("cannot bin an empty dataset")
    if n_bins < 1:
        raise MiscalError(f"n_bins must be >= 1, got {n_bins}")
    for conf in confidences:
        # Guard the library API: the CLI parser already rejects these, but a
        # raw negative confidence would otherwise wrap into the *top* width
        # bin via Python's negative indexing and silently skew every metric.
        if not 0.0 <= conf <= 1.0:
            raise MiscalError(f"confidence {conf!r} is outside [0, 1]; parse it first")


def _make_bin(lower: float, upper: float, members: list[tuple[float, int]]) -> Bin:
    if not members:
        return Bin(lower=lower, upper=upper, count=0, mean_confidence=0.0, accuracy=0.0)
    count = len(members)
    return Bin(
        lower=lower,
        upper=upper,
        count=count,
        mean_confidence=sum(c for c, _ in members) / count,
        accuracy=sum(y for _, y in members) / count,
    )


def equal_width_bins(confidences: list[float], outcomes: list[int], n_bins: int = 10) -> list[Bin]:
    """Split ``[0, 1]`` into ``n_bins`` equal intervals.

    Assignment is half-open ``[lower, upper)`` except the last bin, which is
    closed so a confidence of exactly 1.0 lands in it rather than overflowing.
    """
    _validate(confidences, outcomes, n_bins)
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for conf, outcome in zip(confidences, outcomes):
        index = min(int(conf * n_bins), n_bins - 1)
        buckets[index].append((conf, outcome))
    return [
        _make_bin(i / n_bins, (i + 1) / n_bins, bucket) for i, bucket in enumerate(buckets)
    ]


def equal_mass_bins(confidences: list[float], outcomes: list[int], n_bins: int = 10) -> list[Bin]:
    """Split records into ``n_bins`` groups of near-equal size by confidence.

    Records are sorted by confidence (ties broken by input order, which keeps
    the result deterministic) and chunked; when the dataset is smaller than
    ``n_bins``, only ``len(dataset)`` bins are produced — never empty ones.
    Bin edges are the actual min/max confidence inside each chunk.
    """
    _validate(confidences, outcomes, n_bins)
    pairs = sorted(zip(confidences, outcomes), key=lambda pair: pair[0])
    total = len(pairs)
    n_bins = min(n_bins, total)
    bins: list[Bin] = []
    start = 0
    for i in range(n_bins):
        # Distribute the remainder one record at a time across the first bins.
        size = total // n_bins + (1 if i < total % n_bins else 0)
        chunk = pairs[start : start + size]
        start += size
        bins.append(_make_bin(chunk[0][0], chunk[-1][0], chunk))
    return bins


def make_bins(
    confidences: list[float], outcomes: list[int], n_bins: int = 10, strategy: str = "width"
) -> list[Bin]:
    """Dispatch to the requested binning strategy."""
    if strategy == "width":
        return equal_width_bins(confidences, outcomes, n_bins)
    if strategy == "mass":
        return equal_mass_bins(confidences, outcomes, n_bins)
    raise MiscalError(f"unknown binning strategy {strategy!r}; expected one of {STRATEGIES}")
