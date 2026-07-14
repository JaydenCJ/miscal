"""The SVG renderer: structural validity, determinism, and escaping.

Tests parse the output with ``xml.etree`` rather than regex-matching strings
wherever structure matters, so a markup typo cannot slip through.
"""

import xml.etree.ElementTree as ET

import pytest

from miscal import evaluate, render_reliability_diagram
from miscal.svg import COLOR_ACCURACY, COLOR_GAP
from conftest import hand_dataset

SVG_NS = "{http://www.w3.org/2000/svg}"


@pytest.fixture()
def metrics():
    confs, outs = hand_dataset()
    return evaluate(confs, outs, n_bins=10, strategy="width")


def _render(metrics, **kwargs):
    svg = render_reliability_diagram(metrics, **kwargs)
    return svg, ET.fromstring(svg)


def test_output_is_well_formed_xml_with_requested_dimensions(metrics):
    _, root = _render(metrics, width=800, height=600)
    assert root.tag == f"{SVG_NS}svg"
    assert root.get("width") == "800"
    assert root.get("height") == "600"
    assert root.get("viewBox") == "0 0 800 600"


def test_one_accuracy_and_one_strip_bar_per_occupied_bin(metrics):
    _, root = _render(metrics)
    occupied = sum(1 for b in metrics.bins if b.count > 0)
    bars = [
        el
        for el in root.iter(f"{SVG_NS}rect")
        if el.get("fill") == COLOR_ACCURACY and el.get("fill-opacity") == "0.85"
    ]
    # One bar per occupied bin plus the one legend swatch.
    assert len(bars) == occupied + 1
    strip = [
        el
        for el in root.iter(f"{SVG_NS}rect")
        if el.get("fill") == COLOR_ACCURACY and el.get("fill-opacity") == "0.5"
    ]
    assert len(strip) == occupied


def test_diagonal_reference_line_is_dashed(metrics):
    _, root = _render(metrics)
    dashed = [el for el in root.iter(f"{SVG_NS}line") if el.get("stroke-dasharray")]
    assert len(dashed) == 1


def test_headline_contains_the_key_metrics_and_verdict(metrics):
    svg, _ = _render(metrics)
    assert f"ECE={metrics.ece:.3f}" in svg
    assert f"Brier={metrics.brier:.3f}" in svg
    assert metrics.verdict in svg


def test_title_and_subtitle_are_rendered_and_escaped(metrics):
    svg, root = _render(metrics, title="<run> & \"quotes\"", subtitle="10 width bins")
    assert "10 width bins" in svg
    # Parsing succeeds and the text node round-trips to the original title.
    texts = [el.text for el in root.iter(f"{SVG_NS}text")]
    assert "<run> & \"quotes\"" in texts


def test_rendering_is_deterministic(metrics):
    assert _render(metrics)[0] == _render(metrics)[0]


def test_gap_overlay_marks_miscalibrated_bins(metrics):
    _, root = _render(metrics)
    overlays = [
        el
        for el in root.iter(f"{SVG_NS}rect")
        if el.get("fill") == COLOR_GAP and el.get("fill-opacity") == "0.45"
    ]
    # hand_dataset has three occupied bins, all with a visible gap, plus the
    # legend swatch for the gap color.
    assert len(overlays) == 4


def test_single_record_dataset_renders_without_error():
    metrics = evaluate([0.9], [1], n_bins=10)
    svg, root = _render(metrics)
    assert root.tag == f"{SVG_NS}svg"
    assert "n=1" in svg


def test_equal_mass_bins_render_with_variable_widths():
    confs = [0.1, 0.11, 0.12, 0.9, 0.91, 0.92]
    metrics = evaluate(confs, [0, 0, 0, 1, 1, 1], n_bins=2, strategy="mass")
    _, root = _render(metrics)
    bars = [
        el
        for el in root.iter(f"{SVG_NS}rect")
        if el.get("fill") == COLOR_ACCURACY and el.get("fill-opacity") == "0.85"
    ]
    assert len(bars) == 2 + 1  # two mass bins plus the legend swatch
