"""Command-line interface for miscal.

Subcommands:

* ``report``  — calibration metrics as text or JSON, with optional CI gates.
* ``diagram`` — write a reliability diagram as a standalone SVG file.
* ``compare`` — two runs side by side with signed deltas.
* ``fit``     — fit a temperature and optionally write recalibrated records.

Exit codes: 0 success, 1 a gate failed, 2 bad input or usage. Everything runs
offline; the only I/O is the files named on the command line.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .errors import MiscalError
from .metrics import Metrics, evaluate
from .records import Dataset, FieldMap, load_file
from .report import _n_records, compare_report, json_report, text_report
from .svg import render_reliability_diagram
from .temperature import apply_temperature, fit_temperature

EXIT_OK = 0
EXIT_GATE = 1
EXIT_ERROR = 2


def _add_input_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=("auto", "jsonl", "csv"),
        default="auto",
        help="input format (default: auto — sniff JSONL vs CSV)",
    )
    parser.add_argument("--confidence-field", help="record field holding the confidence")
    parser.add_argument("--correct-field", help="record field holding correctness")
    parser.add_argument("--predicted-field", help="record field holding the predicted label")
    parser.add_argument("--expected-field", help="record field holding the expected label")


def _add_binning_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bins", type=int, default=10, metavar="N", help="number of bins (default: 10)"
    )
    parser.add_argument(
        "--strategy",
        choices=("width", "mass"),
        default="width",
        help="binning strategy: equal-width or equal-mass (default: width)",
    )


def _field_map(args: argparse.Namespace) -> FieldMap:
    return FieldMap(
        confidence=args.confidence_field,
        correct=args.correct_field,
        predicted=args.predicted_field,
        expected=args.expected_field,
    )


def _load(path: str, args: argparse.Namespace) -> Dataset:
    return load_file(path, fmt=args.format, fields=_field_map(args))


def _evaluate(dataset: Dataset, args: argparse.Namespace) -> Metrics:
    return evaluate(
        dataset.confidences, dataset.outcomes, n_bins=args.bins, strategy=args.strategy
    )


def _check_gates(metrics: Metrics, args: argparse.Namespace) -> list[str]:
    """Return a failure message per gate the metrics exceed."""
    failures = []
    gates = (
        ("max_ece", "ECE", metrics.ece),
        ("max_brier", "Brier score", metrics.brier),
        ("max_gap", "absolute confidence gap", abs(metrics.confidence_gap)),
    )
    for attr, label, value in gates:
        limit = getattr(args, attr)
        if limit is not None and value > limit:
            failures.append(f"GATE FAIL: {label} {value:.4f} exceeds limit {limit:.4f}")
    return failures


def _cmd_report(args: argparse.Namespace) -> int:
    metrics = _evaluate(_load(args.file, args), args)
    if args.json:
        sys.stdout.write(json_report(metrics, source=args.file))
    else:
        sys.stdout.write(text_report(metrics, source=args.file))
    failures = _check_gates(metrics, args)
    for failure in failures:
        print(failure, file=sys.stderr)
    return EXIT_GATE if failures else EXIT_OK


def _cmd_diagram(args: argparse.Namespace) -> int:
    metrics = _evaluate(_load(args.file, args), args)
    title = args.title or f"Reliability diagram — {args.file}"
    subtitle = (
        f"{metrics.n_bins} {metrics.strategy} bins · adaptive ECE {metrics.adaptive_ece:.3f}"
        f" · MCE {metrics.mce:.3f}"
    )
    svg = render_reliability_diagram(metrics, title=title, subtitle=subtitle)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(svg)
    print(f"wrote {args.output} ({_n_records(metrics.n_records)}, ECE {metrics.ece:.3f})")
    return EXIT_OK


def _cmd_compare(args: argparse.Namespace) -> int:
    metrics_a = _evaluate(_load(args.file_a, args), args)
    metrics_b = _evaluate(_load(args.file_b, args), args)
    if args.json:
        payload = {
            "a": json.loads(json_report(metrics_a, source=args.file_a)),
            "b": json.loads(json_report(metrics_b, source=args.file_b)),
            "ece_delta": metrics_b.ece - metrics_a.ece,
            "brier_delta": metrics_b.brier - metrics_a.brier,
        }
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(
            compare_report(metrics_a, metrics_b, name_a=args.file_a, name_b=args.file_b)
        )
    return EXIT_OK


def _cmd_fit(args: argparse.Namespace) -> int:
    dataset = _load(args.file, args)
    fit = fit_temperature(
        dataset.confidences, dataset.outcomes, n_bins=args.bins, strategy=args.strategy
    )
    if args.json:
        payload = {
            "temperature": fit.temperature,
            "direction": fit.direction,
            "nll_before": fit.nll_before,
            "nll_after": fit.nll_after,
            "ece_before": fit.ece_before,
            "ece_after": fit.ece_after,
        }
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        print(f"fitted temperature: {fit.temperature:.3f}")
        print(f"  {fit.direction}")
        print(f"  log loss  {fit.nll_before:.4f} -> {fit.nll_after:.4f}")
        print(f"  ECE       {fit.ece_before:.4f} -> {fit.ece_after:.4f}")
    if args.apply:
        rescaled = apply_temperature(dataset.confidences, fit.temperature)
        with open(args.apply, "w", encoding="utf-8") as fh:
            for record, confidence in zip(dataset.records, rescaled):
                obj = dict(record.raw)
                obj["confidence"] = round(confidence, 6)
                obj["correct"] = record.correct
                fh.write(json.dumps(obj, sort_keys=True) + "\n")
        print(f"wrote recalibrated records to {args.apply}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="miscal",
        description="Calibration reports for LLM classifiers: ECE, Brier, reliability diagrams.",
    )
    parser.add_argument("--version", action="version", version=f"miscal {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_report = subparsers.add_parser(
        "report", help="print calibration metrics for a log file"
    )
    p_report.add_argument("file", help="JSONL or CSV log of confidences and outcomes")
    _add_input_options(p_report)
    _add_binning_options(p_report)
    p_report.add_argument("--json", action="store_true", help="emit JSON instead of text")
    p_report.add_argument(
        "--max-ece", type=float, metavar="X", help="exit 1 if ECE exceeds X (CI gate)"
    )
    p_report.add_argument(
        "--max-brier", type=float, metavar="X", help="exit 1 if Brier score exceeds X"
    )
    p_report.add_argument(
        "--max-gap", type=float, metavar="X", help="exit 1 if |confidence gap| exceeds X"
    )
    p_report.set_defaults(func=_cmd_report)

    p_diagram = subparsers.add_parser(
        "diagram", help="write a reliability diagram as SVG"
    )
    p_diagram.add_argument("file", help="JSONL or CSV log of confidences and outcomes")
    p_diagram.add_argument(
        "-o", "--output", default="reliability.svg", help="output SVG path"
    )
    p_diagram.add_argument("--title", help="diagram title (default: derived from file name)")
    _add_input_options(p_diagram)
    _add_binning_options(p_diagram)
    p_diagram.set_defaults(func=_cmd_diagram)

    p_compare = subparsers.add_parser(
        "compare", help="compare calibration between two log files"
    )
    p_compare.add_argument("file_a", help="baseline log file")
    p_compare.add_argument("file_b", help="candidate log file")
    _add_input_options(p_compare)
    _add_binning_options(p_compare)
    p_compare.add_argument("--json", action="store_true", help="emit JSON instead of text")
    p_compare.set_defaults(func=_cmd_compare)

    p_fit = subparsers.add_parser(
        "fit", help="fit a temperature that recalibrates the confidences"
    )
    p_fit.add_argument("file", help="JSONL or CSV log of confidences and outcomes")
    _add_input_options(p_fit)
    _add_binning_options(p_fit)
    p_fit.add_argument("--json", action="store_true", help="emit JSON instead of text")
    p_fit.add_argument(
        "--apply", metavar="OUT.jsonl", help="write records with recalibrated confidences"
    )
    p_fit.set_defaults(func=_cmd_fit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except MiscalError as exc:
        print(f"miscal: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except OSError as exc:
        print(f"miscal: error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
