# Contributing to miscal

Thanks for your interest in contributing. Issues, discussions, and pull
requests are all welcome.

## Development setup

```bash
git clone https://github.com/JaydenCJ/miscal
cd miscal
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the checks

```bash
pytest                 # 93 unit + CLI tests, fully offline
bash scripts/smoke.sh  # end-to-end smoke: report, diagram, compare, fit, gates
```

Both must pass before a pull request is reviewed; the smoke script must print
`SMOKE OK`. The whole suite runs offline in a few seconds and needs no API
keys or model access.

## Ground rules

- **No new runtime dependencies.** The package is standard-library only; that
  is a feature, not an accident. Test-only dependencies belong in the `dev`
  extra and need justification in the PR.
- **Metric changes need hand-computed references.** Anything touching ECE,
  Brier, log loss, or binning must update the worked-out reference values in
  `tests/conftest.py` and explain the arithmetic in the PR description.
- **Parsing rules are documented contracts.** Changing how a confidence
  string or number is interpreted requires updating the reference tables in
  all three READMEs and `docs/record-format.md` in the same pull request.
- **Keep the three READMEs aligned.** `README.md`, `README.zh.md`, and
  `README.ja.md` share the same line-for-line structure; update all three
  when you change one (English is the authoritative version).
- Code comments and doc comments are written in English.

## Reporting bugs

Please include `miscal --version` output, a few (redacted) lines of the input
log that triggers the problem, and the full command line. Metric
discrepancies are easiest to act on when you show the expected value and how
you computed it.

## Security

Please do not open public issues for security problems; use GitHub private
vulnerability reporting on this repository instead.
