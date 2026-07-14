"""End-to-end CLI behavior through ``miscal.cli.main`` — no subprocesses.

Covers exit codes (0 success, 1 gate failure, 2 bad input), JSON output,
field-mapping flags, and the artifacts each subcommand writes.
"""

import json
import xml.etree.ElementTree as ET

import pytest

from miscal import __version__
from miscal.cli import main
from conftest import write_jsonl


@pytest.fixture()
def overconfident_log(tmp_path):
    rows = (
        [{"confidence": 0.95, "correct": True} for _ in range(6)]
        + [{"confidence": 0.95, "correct": False} for _ in range(4)]
        + [{"confidence": 0.6, "correct": True} for _ in range(3)]
        + [{"confidence": 0.6, "correct": False} for _ in range(3)]
    )
    return write_jsonl(tmp_path / "run.jsonl", rows)


def test_report_prints_text_and_exits_zero(overconfident_log, capsys):
    assert main(["report", str(overconfident_log)]) == 0
    out = capsys.readouterr().out
    assert "miscal report" in out
    assert "ECE" in out
    assert "verdict: overconfident" in out


def test_report_json_is_machine_readable(overconfident_log, capsys):
    assert main(["report", str(overconfident_log), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_records"] == 16
    assert 0.0 <= payload["ece"] <= 1.0


def test_bad_inputs_exit_two_with_useful_errors(tmp_path, capsys):
    assert main(["report", "no-such-file.jsonl"]) == 2
    assert "miscal: error:" in capsys.readouterr().err
    path = tmp_path / "bad.jsonl"
    path.write_text('{"confidence": 0.9, "correct": true}\n{"confidence": "meh", "correct": true}\n')
    assert main(["report", str(path)]) == 2
    assert "line 2" in capsys.readouterr().err


def test_ece_gate_passes_under_and_fails_over_the_limit(overconfident_log, capsys):
    assert main(["report", str(overconfident_log), "--max-ece", "0.9"]) == 0
    assert "GATE FAIL" not in capsys.readouterr().err
    assert main(["report", str(overconfident_log), "--max-ece", "0.01"]) == 1
    err = capsys.readouterr().err
    assert "GATE FAIL: ECE" in err
    assert "exceeds limit 0.0100" in err


def test_gap_and_brier_gates_report_each_failure(overconfident_log, capsys):
    code = main(
        ["report", str(overconfident_log), "--max-gap", "0.01", "--max-brier", "0.01"]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "absolute confidence gap" in err
    assert "Brier score" in err


def test_diagram_writes_a_valid_svg(overconfident_log, tmp_path, capsys):
    out = tmp_path / "reliability.svg"
    assert main(["diagram", str(overconfident_log), "-o", str(out)]) == 0
    root = ET.parse(out).getroot()
    assert root.tag.endswith("svg")
    assert "wrote" in capsys.readouterr().out


def test_diagram_honors_title_and_bins(overconfident_log, tmp_path):
    out = tmp_path / "custom.svg"
    main(["diagram", str(overconfident_log), "-o", str(out), "--title", "Nightly eval", "--bins", "5"])
    svg = out.read_text(encoding="utf-8")
    assert "Nightly eval" in svg
    assert "5 width bins" in svg


def test_compare_prints_both_sources(tmp_path, capsys):
    a = write_jsonl(tmp_path / "a.jsonl", [{"confidence": 0.9, "correct": False}] * 4)
    b = write_jsonl(tmp_path / "b.jsonl", [{"confidence": 0.9, "correct": True}] * 4)
    assert main(["compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "A: " in out and "a.jsonl" in out
    assert "B: " in out and "b.jsonl" in out
    assert "(B better)" in out


def test_compare_json_includes_deltas(tmp_path, capsys):
    a = write_jsonl(tmp_path / "a.jsonl", [{"confidence": 0.9, "correct": False}] * 4)
    b = write_jsonl(tmp_path / "b.jsonl", [{"confidence": 0.9, "correct": True}] * 4)
    assert main(["compare", str(a), str(b), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ece_delta"] == pytest.approx(payload["b"]["ece"] - payload["a"]["ece"])
    assert payload["a"]["verdict"] == "overconfident"


def test_fit_prints_temperature_in_text_and_json(overconfident_log, capsys):
    assert main(["fit", str(overconfident_log)]) == 0
    out = capsys.readouterr().out
    assert "fitted temperature:" in out
    assert "overconfident" in out
    assert "log loss" in out
    assert main(["fit", str(overconfident_log), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["temperature"] > 1.0
    assert payload["nll_after"] <= payload["nll_before"]


def test_fit_apply_writes_recalibrated_records(overconfident_log, tmp_path, capsys):
    out = tmp_path / "recalibrated.jsonl"
    assert main(["fit", str(overconfident_log), "--apply", str(out)]) == 0
    lines = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 16
    assert all(0.0 <= row["confidence"] <= 1.0 for row in lines)
    # The recalibrated file is itself a valid miscal input.
    assert main(["report", str(out)]) == 0


def test_csv_input_end_to_end(tmp_path, capsys):
    path = tmp_path / "run.csv"
    path.write_text(
        "id,confidence,predicted,expected\n"
        "1,90%,spam,spam\n"
        "2,very likely,ham,spam\n"
        "3,0.6,spam,spam\n",
        encoding="utf-8",
    )
    assert main(["report", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_records"] == 3
    assert payload["accuracy"] == pytest.approx(2 / 3)


def test_field_mapping_flags(tmp_path, capsys):
    path = write_jsonl(
        tmp_path / "custom.jsonl",
        [{"certainty": "80%", "was_right": "yes"}, {"certainty": 0.5, "was_right": "no"}],
    )
    code = main(
        [
            "report",
            str(path),
            "--json",
            "--confidence-field",
            "certainty",
            "--correct-field",
            "was_right",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mean_confidence"] == pytest.approx(0.65)


def test_strategy_flag_switches_to_equal_mass_bins(overconfident_log, capsys):
    assert main(["report", str(overconfident_log), "--strategy", "mass", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["strategy"] == "mass"
    assert all(b["count"] > 0 for b in payload["bins"])


def test_version_flag_and_missing_subcommand(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"miscal {__version__}"
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
