#!/usr/bin/env bash
# Smoke test for miscal: run the real CLI end-to-end against the committed
# example logs — report, JSON output, SVG diagram, comparison, temperature
# fit, and the CI gate exit codes.
# Self-contained: pure stdlib, no network, idempotent (works from a clean tree).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
if [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
fi

# The package has zero runtime dependencies, so running from src/ needs no install.
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/miscal-smoke.XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT

fail() { echo "SMOKE FAIL: $1" >&2; exit 1; }

echo "[smoke] python: $("$PYTHON" --version 2>&1)"
SAMPLE="$ROOT/examples/sample_run.jsonl"
SAMPLE_V2="$ROOT/examples/sample_run_v2.jsonl"

# 1. report: text output with metrics and the overconfident verdict.
report_out="$("$PYTHON" -m miscal report "$SAMPLE")" || fail "report exited non-zero"
echo "$report_out" | sed 's/^/[report] /' | head -8
echo "$report_out" | grep -q "records: 200" || fail "report missing record count"
echo "$report_out" | grep -q "ECE                0.146" || fail "report ECE drifted"
echo "$report_out" | grep -q "verdict: overconfident" || fail "report verdict missing"

# 2. report --json: machine-readable output with the same numbers.
json_out="$("$PYTHON" -m miscal report "$SAMPLE" --json)"
echo "$json_out" | "$PYTHON" -c '
import json, sys
payload = json.load(sys.stdin)
assert payload["n_records"] == 200, payload["n_records"]
assert abs(payload["ece"] - 0.146) < 0.001, payload["ece"]
assert payload["verdict"] == "overconfident"
' || fail "JSON report is wrong or unparseable"

# 3. diagram: writes a well-formed SVG containing the headline metrics.
"$PYTHON" -m miscal diagram "$SAMPLE" -o "$WORKDIR/reliability.svg" >/dev/null \
  || fail "diagram exited non-zero"
[ -s "$WORKDIR/reliability.svg" ] || fail "diagram wrote no file"
"$PYTHON" -c "import xml.etree.ElementTree as ET; ET.parse('$WORKDIR/reliability.svg')" \
  || fail "diagram SVG is not well-formed XML"
grep -q "ECE=0.146" "$WORKDIR/reliability.svg" || fail "diagram missing ECE headline"

# 4. compare: the prompt-fixed run must come out better calibrated.
compare_out="$("$PYTHON" -m miscal compare "$SAMPLE" "$SAMPLE_V2")" \
  || fail "compare exited non-zero"
echo "$compare_out" | sed 's/^/[compare] /'
echo "$compare_out" | grep -q "(B better)" || fail "compare did not mark v2 as better"
echo "$compare_out" | grep -q "verdict B: well-calibrated" || fail "compare verdict B wrong"

# 5. fit: temperature > 1 for an overconfident run; --apply round-trips.
fit_out="$("$PYTHON" -m miscal fit "$SAMPLE" --apply "$WORKDIR/recalibrated.jsonl")" \
  || fail "fit exited non-zero"
echo "$fit_out" | sed 's/^/[fit] /'
echo "$fit_out" | grep -q "fitted temperature: 5.227" || fail "fitted temperature drifted"
echo "$fit_out" | grep -q "overconfident" || fail "fit direction missing"
[ -s "$WORKDIR/recalibrated.jsonl" ] || fail "fit --apply wrote no file"
"$PYTHON" -m miscal report "$WORKDIR/recalibrated.jsonl" >/dev/null \
  || fail "recalibrated output is not a valid miscal input"

# 6. CI gate: exit 1 when ECE exceeds the limit, 0 when it passes.
set +e
"$PYTHON" -m miscal report "$SAMPLE" --max-ece 0.10 >/dev/null 2>"$WORKDIR/gate.err"
gate_rc=$?
set -e
[ "$gate_rc" -eq 1 ] || fail "gate on overconfident run should exit 1, got $gate_rc"
grep -q "GATE FAIL: ECE" "$WORKDIR/gate.err" || fail "gate failure message missing"
"$PYTHON" -m miscal report "$SAMPLE_V2" --max-ece 0.10 >/dev/null \
  || fail "gate on calibrated run should exit 0"
echo "[gate] exit 1 on ECE 0.146 > 0.10, exit 0 on the fixed run"

# 7. Bad input exits 2; --version agrees with the package version.
set +e
"$PYTHON" -m miscal report "$WORKDIR/does-not-exist.jsonl" >/dev/null 2>&1
missing_rc=$?
set -e
[ "$missing_rc" -eq 2 ] || fail "missing file should exit 2, got $missing_rc"
version_out="$("$PYTHON" -m miscal --version)"
pkg_version="$("$PYTHON" -c 'import miscal; print(miscal.__version__)')"
[ "$version_out" = "miscal $pkg_version" ] \
  || fail "--version mismatch: '$version_out' vs package '$pkg_version'"

echo "SMOKE OK"
