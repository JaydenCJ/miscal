"""Render reliability diagrams as standalone SVG — no matplotlib, no fonts.

The output is a single self-contained ``<svg>`` document using only system
font stacks and inline styles, so it drops straight into a README, a Slack
message, or a static site. Rendering is fully deterministic: the same
metrics always produce byte-identical SVG.

Layout (top to bottom): title + headline metrics, the reliability plot
(stated confidence on x, observed accuracy on y, dashed diagonal = perfect
calibration), and a sample-count strip showing how many records back each
bin — a bar touching the diagonal means little if it holds three records.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from .metrics import Metrics

# A colorblind-safe pair: blue for observed accuracy, warm red for the
# miscalibration gap. The diagonal and axes stay neutral gray.
COLOR_ACCURACY = "#4C78A8"
COLOR_GAP = "#E45756"
COLOR_AXIS = "#6b6b6b"
COLOR_GRID = "#d9d9d9"
COLOR_TEXT = "#222222"
FONT = "font-family='-apple-system, Segoe UI, Helvetica, Arial, sans-serif'"


def _fmt(value: float) -> str:
    """Format a coordinate with enough precision to be exact at 1x zoom."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def render_reliability_diagram(
    metrics: Metrics,
    title: str = "Reliability diagram",
    subtitle: str | None = None,
    width: int = 640,
    height: int = 520,
) -> str:
    """Render ``metrics`` as a reliability diagram, returned as SVG text."""
    left, right, top = 64.0, 24.0, 78.0
    strip_h = 56.0
    bottom = strip_h + 56.0
    plot_w = width - left - right
    plot_h = height - top - bottom
    strip_top = top + plot_h + 34.0

    def px(conf: float) -> float:
        return left + conf * plot_w

    def py(acc: float) -> float:
        return top + (1.0 - acc) * plot_h

    parts: list[str] = []
    parts.append(
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}' role='img' aria-label='{escape(title)}'>"
    )
    parts.append(f"<rect width='{width}' height='{height}' fill='#ffffff'/>")

    # Title block and headline numbers.
    parts.append(
        f"<text x='{left}' y='28' font-size='17' font-weight='600' {FONT} "
        f"fill='{COLOR_TEXT}'>{escape(title)}</text>"
    )
    headline = (
        f"n={metrics.n_records}  ECE={metrics.ece:.3f}  Brier={metrics.brier:.3f}  "
        f"acc={metrics.accuracy:.3f}  conf={metrics.mean_confidence:.3f}  "
        f"verdict: {metrics.verdict}"
    )
    parts.append(
        f"<text x='{left}' y='48' font-size='12' {FONT} fill='{COLOR_AXIS}'>"
        f"{escape(headline)}</text>"
    )
    if subtitle:
        parts.append(
            f"<text x='{left}' y='64' font-size='11' {FONT} fill='{COLOR_AXIS}'>"
            f"{escape(subtitle)}</text>"
        )

    # Gridlines and axis ticks at 0.0, 0.2, ..., 1.0 on both axes.
    for i in range(6):
        v = i / 5.0
        gx, gy = px(v), py(v)
        parts.append(
            f"<line x1='{_fmt(gx)}' y1='{_fmt(top)}' x2='{_fmt(gx)}' "
            f"y2='{_fmt(top + plot_h)}' stroke='{COLOR_GRID}' stroke-width='1'/>"
        )
        parts.append(
            f"<line x1='{_fmt(left)}' y1='{_fmt(gy)}' x2='{_fmt(left + plot_w)}' "
            f"y2='{_fmt(gy)}' stroke='{COLOR_GRID}' stroke-width='1'/>"
        )
        parts.append(
            f"<text x='{_fmt(gx)}' y='{_fmt(top + plot_h + 16)}' font-size='11' {FONT} "
            f"fill='{COLOR_AXIS}' text-anchor='middle'>{v:.1f}</text>"
        )
        parts.append(
            f"<text x='{_fmt(left - 8)}' y='{_fmt(gy + 4)}' font-size='11' {FONT} "
            f"fill='{COLOR_AXIS}' text-anchor='end'>{v:.1f}</text>"
        )

    # Bars: observed accuracy in blue, the gap up/down to stated confidence in red.
    max_count = max((b.count for b in metrics.bins), default=0)
    for b in metrics.bins:
        if b.count == 0:
            continue
        x0 = px(b.lower) + 1.0
        bar_w = max(px(b.upper) - px(b.lower) - 2.0, 1.5)
        acc_y = py(b.accuracy)
        parts.append(
            f"<rect x='{_fmt(x0)}' y='{_fmt(acc_y)}' width='{_fmt(bar_w)}' "
            f"height='{_fmt(py(0.0) - acc_y)}' fill='{COLOR_ACCURACY}' fill-opacity='0.85'/>"
        )
        conf_y = py(b.mean_confidence)
        gap_top, gap_bottom = min(acc_y, conf_y), max(acc_y, conf_y)
        if gap_bottom - gap_top > 0.5:
            parts.append(
                f"<rect x='{_fmt(x0)}' y='{_fmt(gap_top)}' width='{_fmt(bar_w)}' "
                f"height='{_fmt(gap_bottom - gap_top)}' fill='{COLOR_GAP}' "
                f"fill-opacity='0.45'/>"
            )
        parts.append(
            f"<line x1='{_fmt(x0)}' y1='{_fmt(conf_y)}' x2='{_fmt(x0 + bar_w)}' "
            f"y2='{_fmt(conf_y)}' stroke='{COLOR_GAP}' stroke-width='2'/>"
        )

    # Perfect-calibration diagonal, drawn above the bars for readability.
    parts.append(
        f"<line x1='{_fmt(px(0.0))}' y1='{_fmt(py(0.0))}' x2='{_fmt(px(1.0))}' "
        f"y2='{_fmt(py(1.0))}' stroke='{COLOR_AXIS}' stroke-width='1.5' "
        f"stroke-dasharray='6 4'/>"
    )

    # Plot frame and axis labels.
    parts.append(
        f"<rect x='{_fmt(left)}' y='{_fmt(top)}' width='{_fmt(plot_w)}' "
        f"height='{_fmt(plot_h)}' fill='none' stroke='{COLOR_AXIS}' stroke-width='1'/>"
    )
    parts.append(
        f"<text x='{_fmt(left + plot_w / 2)}' y='{_fmt(height - 10)}' font-size='12' "
        f"{FONT} fill='{COLOR_TEXT}' text-anchor='middle'>stated confidence</text>"
    )
    parts.append(
        f"<text x='16' y='{_fmt(top + plot_h / 2)}' font-size='12' {FONT} "
        f"fill='{COLOR_TEXT}' text-anchor='middle' "
        f"transform='rotate(-90 16 {_fmt(top + plot_h / 2)})'>observed accuracy</text>"
    )

    # Sample-count strip under the plot.
    parts.append(
        f"<text x='{_fmt(left)}' y='{_fmt(strip_top - 6)}' font-size='10' {FONT} "
        f"fill='{COLOR_AXIS}'>records per bin (max {max_count})</text>"
    )
    for b in metrics.bins:
        if b.count == 0 or max_count == 0:
            continue
        x0 = px(b.lower) + 1.0
        bar_w = max(px(b.upper) - px(b.lower) - 2.0, 1.5)
        bar_h = max(strip_h * b.count / max_count, 1.0)
        parts.append(
            f"<rect x='{_fmt(x0)}' y='{_fmt(strip_top + strip_h - bar_h)}' "
            f"width='{_fmt(bar_w)}' height='{_fmt(bar_h)}' fill='{COLOR_ACCURACY}' "
            f"fill-opacity='0.5'/>"
        )
    parts.append(
        f"<line x1='{_fmt(left)}' y1='{_fmt(strip_top + strip_h)}' "
        f"x2='{_fmt(left + plot_w)}' y2='{_fmt(strip_top + strip_h)}' "
        f"stroke='{COLOR_AXIS}' stroke-width='1'/>"
    )

    # Legend, kept inside the plot's top-left corner.
    lx, ly = left + 10.0, top + 14.0
    parts.append(
        f"<rect x='{_fmt(lx)}' y='{_fmt(ly - 9)}' width='12' height='12' "
        f"fill='{COLOR_ACCURACY}' fill-opacity='0.85'/>"
    )
    parts.append(
        f"<text x='{_fmt(lx + 18)}' y='{_fmt(ly + 1)}' font-size='11' {FONT} "
        f"fill='{COLOR_TEXT}'>observed accuracy</text>"
    )
    parts.append(
        f"<rect x='{_fmt(lx)}' y='{_fmt(ly + 9)}' width='12' height='12' "
        f"fill='{COLOR_GAP}' fill-opacity='0.45'/>"
    )
    parts.append(
        f"<text x='{_fmt(lx + 18)}' y='{_fmt(ly + 19)}' font-size='11' {FONT} "
        f"fill='{COLOR_TEXT}'>calibration gap</text>"
    )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"
