"""The README quickstart runs verbatim against the committed example log.

If these numbers drift, the captured output shown in all three READMEs is
stale — regenerate it (or the sample data) rather than loosening the test.
"""

import json
import xml.etree.ElementTree as ET

import pytest

from miscal.cli import main


def test_quickstart_report_matches_the_captured_output(sample_run_path, capsys):
    assert main(["report", str(sample_run_path)]) == 0
    out = capsys.readouterr().out
    assert "records: 200   bins: 10 (width)" in out
    assert "accuracy           0.665" in out
    assert "mean confidence    0.809" in out
    assert "ECE                0.146" in out
    assert "Brier score        0.249" in out
    assert "verdict: overconfident (stated confidence exceeds accuracy by 14.4 points)" in out


def test_quickstart_fit_matches_the_captured_output(sample_run_path, capsys):
    assert main(["fit", str(sample_run_path)]) == 0
    out = capsys.readouterr().out
    assert "fitted temperature: 5.227" in out
    assert "ECE       0.1462 -> 0.0889" in out


def test_quickstart_diagram_produces_the_committed_hero_image(sample_run_path, tmp_path):
    out = tmp_path / "reliability.svg"
    assert main(["diagram", str(sample_run_path), "-o", str(out)]) == 0
    root = ET.parse(out).getroot()
    assert root.tag.endswith("svg")


def test_hero_image_is_byte_identical_to_real_diagram_output(tmp_path, monkeypatch):
    # docs/assets/demo.svg must be exactly what the README's diagram command
    # produces (same relative path in the title), or the hero image is a lie.
    from conftest import REPO_ROOT

    monkeypatch.chdir(REPO_ROOT)
    out = tmp_path / "demo.svg"
    assert main(["diagram", "examples/sample_run.jsonl", "-o", str(out)]) == 0
    committed = (REPO_ROOT / "docs" / "assets" / "demo.svg").read_text(encoding="utf-8")
    assert out.read_text(encoding="utf-8") == committed


def test_compare_gate_story_from_the_readme(sample_run_path, sample_run_v2_path, capsys):
    # The README claims run v2 (after the prompt fix) is better calibrated.
    assert main(["compare", str(sample_run_path), str(sample_run_v2_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ece_delta"] < 0
    assert payload["b"]["verdict"] == "well-calibrated"
    # And that v2 passes the CI gate the original fails.
    assert main(["report", str(sample_run_v2_path), "--max-ece", "0.10"]) == 0
    assert main(["report", str(sample_run_path), "--max-ece", "0.10"]) == 1


def test_sample_log_mixes_confidence_representations(sample_run_path):
    text = sample_run_path.read_text(encoding="utf-8")
    lines = [json.loads(ln) for ln in text.splitlines()]
    kinds = {type(row["confidence"]).__name__ for row in lines}
    assert kinds == {"float", "str"}
    assert any(str(row["confidence"]).endswith("%") for row in lines)
    assert any("/" in str(row["confidence"]) for row in lines)
