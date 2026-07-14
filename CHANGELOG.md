# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-13

### Added

- JSONL and CSV record loading with field-alias resolution (`confidence`,
  `conf`, `p`, `prob`, …), explicit field-mapping flags, correctness derived
  from `predicted`/`expected` label pairs, and parse errors that carry the
  exact line number of the offending record.
- Verbalized-confidence parsing: percentages (`"85%"`), fractions (`"9/10"`),
  numeric strings, a documented 0–100 percent rule for bare numbers, and a
  33-entry anchor-word scale (`"very likely"` → 0.90) for prose confidences.
- Calibration metrics in pure stdlib: ECE (equal-width), adaptive ECE
  (equal-mass), MCE, Brier score with the Murphy reliability / resolution /
  uncertainty decomposition, log loss with epsilon clamping, signed
  confidence gap, and a plain-English verdict.
- Two binning strategies (`width`, `mass`) with deterministic boundary
  handling, shared by the metrics, the reports, and the diagram.
- Reliability diagrams as standalone, deterministic SVG: accuracy bars,
  calibration-gap overlays, perfect-calibration diagonal, per-bin sample
  counts, and headline metrics — no matplotlib, no fonts, no network.
- Temperature scaling: `fit` finds the NLL-minimizing temperature with a
  golden-section search and reports before/after log loss and ECE;
  `--apply` writes recalibrated records that are valid miscal input.
- `miscal` CLI: `report` (text/JSON, with `--max-ece`, `--max-brier`,
  `--max-gap` CI gates and exit code 1 on breach), `diagram`, `compare`
  (side-by-side deltas), and `fit`.
- Runnable example: a seeded overconfident intent-classifier log plus its
  prompt-fixed counterpart, regenerable byte-for-byte with
  `examples/make_sample.py`.
- 93 deterministic offline tests and `scripts/smoke.sh` exercising the real
  CLI end-to-end.

### Notes

- The repository ships no CI workflow; verification is local — `pip install -e '.[dev]' && pytest && bash scripts/smoke.sh`.

[0.1.0]: https://github.com/JaydenCJ/miscal/releases/tag/v0.1.0
