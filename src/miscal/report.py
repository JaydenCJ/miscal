"""Turn a :class:`~miscal.metrics.Metrics` into text, JSON, and comparisons.

The text report is designed for terminals and code-review comments: fixed
layout, aligned numbers, a per-bin table, and a one-line verdict. The JSON
report is the same data, machine-readable, for dashboards and CI artifacts.
"""

from __future__ import annotations

import json
from typing import Optional

from .metrics import Metrics

# Metric rows shown in reports and comparisons: (attribute, label, lower_is_better)
_METRIC_ROWS = (
    ("accuracy", "accuracy", None),
    ("mean_confidence", "mean confidence", None),
    ("confidence_gap", "confidence gap", None),
    ("ece", "ECE", True),
    ("adaptive_ece", "adaptive ECE", True),
    ("mce", "MCE", True),
    ("brier", "Brier score", True),
    ("log_loss", "log loss", True),
)


def _signed(value: float) -> str:
    return f"{value:+.3f}"


def _n_records(n: int) -> str:
    return f"{n} record" if n == 1 else f"{n} records"


def text_report(metrics: Metrics, source: Optional[str] = None) -> str:
    """Render the full human-readable calibration report."""
    lines: list[str] = []
    header = "miscal report"
    if source:
        header += f" — {source}"
    lines.append(header)
    lines.append(
        f"records: {metrics.n_records}   bins: {metrics.n_bins} ({metrics.strategy})"
    )
    lines.append("")
    lines.append(f"  accuracy           {metrics.accuracy:.3f}")
    lines.append(f"  mean confidence    {metrics.mean_confidence:.3f}")
    lines.append(f"  confidence gap     {_signed(metrics.confidence_gap)}")
    lines.append(f"  ECE                {metrics.ece:.3f}")
    lines.append(f"  adaptive ECE       {metrics.adaptive_ece:.3f}")
    lines.append(f"  MCE                {metrics.mce:.3f}")
    lines.append(f"  Brier score        {metrics.brier:.3f}")
    lines.append(f"    reliability      {metrics.brier_reliability:.3f}")
    lines.append(f"    resolution       {metrics.brier_resolution:.3f}")
    lines.append(f"    uncertainty      {metrics.brier_uncertainty:.3f}")
    lines.append(f"  log loss           {metrics.log_loss:.3f}")
    lines.append("")
    lines.append("  bin        n     conf    acc     gap")
    for b in metrics.bins:
        span = f"[{b.lower:.2f},{b.upper:.2f}]"
        if b.count == 0:
            lines.append(f"  {span:<10} {0:>4}       -      -       -")
        else:
            lines.append(
                f"  {span:<10} {b.count:>4}   {b.mean_confidence:.3f}  {b.accuracy:.3f}  "
                f"{_signed(b.gap)}"
            )
    lines.append("")
    lines.append(f"verdict: {metrics.verdict}{_verdict_detail(metrics)}")
    return "\n".join(lines) + "\n"


def _verdict_detail(metrics: Metrics) -> str:
    gap_points = abs(metrics.confidence_gap) * 100.0
    if metrics.verdict == "overconfident":
        return f" (stated confidence exceeds accuracy by {gap_points:.1f} points)"
    if metrics.verdict == "underconfident":
        return f" (accuracy exceeds stated confidence by {gap_points:.1f} points)"
    return f" (confidence within {gap_points:.1f} points of accuracy)"


def json_report(metrics: Metrics, source: Optional[str] = None) -> str:
    """Render metrics as pretty-printed JSON with sorted keys (diff-friendly)."""
    payload = {
        "source": source,
        "n_records": metrics.n_records,
        "n_bins": metrics.n_bins,
        "strategy": metrics.strategy,
        "accuracy": metrics.accuracy,
        "mean_confidence": metrics.mean_confidence,
        "confidence_gap": metrics.confidence_gap,
        "ece": metrics.ece,
        "adaptive_ece": metrics.adaptive_ece,
        "mce": metrics.mce,
        "brier": metrics.brier,
        "brier_reliability": metrics.brier_reliability,
        "brier_resolution": metrics.brier_resolution,
        "brier_uncertainty": metrics.brier_uncertainty,
        "log_loss": metrics.log_loss,
        "verdict": metrics.verdict,
        "bins": [
            {
                "lower": b.lower,
                "upper": b.upper,
                "count": b.count,
                "mean_confidence": b.mean_confidence,
                "accuracy": b.accuracy,
                "gap": b.gap,
            }
            for b in metrics.bins
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def compare_report(
    a: Metrics, b: Metrics, name_a: str = "a", name_b: str = "b"
) -> str:
    """Render a side-by-side comparison of two runs with signed deltas.

    For the error metrics (ECE, MCE, Brier, log loss) a negative delta means
    run B is better calibrated than run A; the trailing marker says which.
    """
    width = 8
    lines = [
        "miscal compare",
        f"  A: {name_a} ({_n_records(a.n_records)})",
        f"  B: {name_b} ({_n_records(b.n_records)})",
        "",
        f"  {'metric':<17} {'A':>{width}} {'B':>{width}}   delta",
    ]
    for attr, label, lower_is_better in _METRIC_ROWS:
        va, vb = getattr(a, attr), getattr(b, attr)
        delta = vb - va
        marker = ""
        if lower_is_better and abs(delta) >= 0.0005:
            marker = "  (B better)" if delta < 0 else "  (B worse)"
        lines.append(
            f"  {label:<17} {va:>{width}.3f} {vb:>{width}.3f}  {_signed(delta)}{marker}"
        )
    lines.append("")
    lines.append(f"  verdict A: {a.verdict}")
    lines.append(f"  verdict B: {b.verdict}")
    return "\n".join(lines) + "\n"
