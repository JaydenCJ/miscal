"""Load logged classifier outcomes from JSONL or CSV files.

A *record* is one classifier decision: a confidence (however the model
expressed it — see :mod:`miscal.verbal`) plus whether the decision was
correct. Correctness is either logged directly (``"correct": true``) or
derived by comparing a predicted label against the expected one.

Field names are resolved through alias lists so common logging schemas work
untouched; explicit ``FieldMap`` overrides win over every alias.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .errors import EmptyDatasetError, MiscalError, RecordError
from .verbal import parse_confidence

CONFIDENCE_ALIASES = ("confidence", "conf", "p", "prob", "probability", "score")
CORRECT_ALIASES = ("correct", "is_correct", "hit", "success")
PREDICTED_ALIASES = ("predicted", "prediction", "pred", "output")
EXPECTED_ALIASES = ("expected", "gold", "label", "truth", "answer")

_TRUE_STRINGS = frozenset({"true", "t", "yes", "y", "1"})
_FALSE_STRINGS = frozenset({"false", "f", "no", "n", "0"})


@dataclass(frozen=True)
class FieldMap:
    """Explicit field names, overriding the built-in aliases when set."""

    confidence: str | None = None
    correct: str | None = None
    predicted: str | None = None
    expected: str | None = None


@dataclass(frozen=True)
class Record:
    """One parsed classifier decision."""

    confidence: float
    correct: bool
    raw: dict = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class Dataset:
    """A parsed set of records plus the vectors the metrics layer consumes."""

    records: tuple[Record, ...]

    @property
    def confidences(self) -> list[float]:
        return [r.confidence for r in self.records]

    @property
    def outcomes(self) -> list[int]:
        return [1 if r.correct else 0 for r in self.records]

    def __len__(self) -> int:
        return len(self.records)


def _lookup(obj: dict, explicit: str | None, aliases: Sequence[str], line: int):
    """Fetch a field by explicit name or alias list; (found, value) tuple."""
    if explicit is not None:
        if explicit not in obj:
            raise RecordError(f"field {explicit!r} not present in record", line)
        return True, obj[explicit]
    for name in aliases:
        if name in obj:
            return True, obj[name]
    return False, None


def _parse_correct(value: object, line: int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
    raise RecordError(f"cannot interpret correctness value {value!r} as a boolean", line)


def record_from_dict(obj: dict, fields: FieldMap = FieldMap(), line: int = 0) -> Record:
    """Build a :class:`Record` from one parsed JSON object or CSV row."""
    found, raw_conf = _lookup(obj, fields.confidence, CONFIDENCE_ALIASES, line)
    if not found or raw_conf is None or raw_conf == "":
        raise RecordError(
            f"no confidence field found (looked for {', '.join(CONFIDENCE_ALIASES)})", line
        )
    try:
        confidence = parse_confidence(raw_conf)
    except MiscalError as exc:
        raise RecordError(str(exc), line) from exc

    found, raw_correct = _lookup(obj, fields.correct, CORRECT_ALIASES, line)
    if found:
        correct = _parse_correct(raw_correct, line)
    else:
        has_pred, predicted = _lookup(obj, fields.predicted, PREDICTED_ALIASES, line)
        has_gold, expected = _lookup(obj, fields.expected, EXPECTED_ALIASES, line)
        if not (has_pred and has_gold):
            raise RecordError(
                "record has neither a correctness field nor a predicted/expected pair", line
            )
        # Labels compare as trimmed, case-insensitive strings: LLM outputs are
        # text, and "Spam" vs "spam" is a formatting quirk, not a miss.
        correct = str(predicted).strip().lower() == str(expected).strip().lower()

    return Record(confidence=confidence, correct=correct, raw=dict(obj))


def _iter_jsonl(text: str, fields: FieldMap) -> Iterable[Record]:
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise RecordError(f"invalid JSON ({exc.msg})", line_no) from exc
        if not isinstance(obj, dict):
            raise RecordError("each JSONL line must be an object", line_no)
        yield record_from_dict(obj, fields, line_no)


def _iter_csv(text: str, fields: FieldMap) -> Iterable[Record]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return
    for line_no, row in enumerate(reader, start=2):  # line 1 is the header
        if all((v is None or str(v).strip() == "") for v in row.values()):
            continue
        yield record_from_dict({k: v for k, v in row.items() if k is not None}, fields, line_no)


def parse_text(text: str, fmt: str = "auto", fields: FieldMap = FieldMap()) -> Dataset:
    """Parse log text in ``jsonl`` or ``csv`` format into a :class:`Dataset`.

    ``fmt="auto"`` sniffs the first non-blank line: a leading ``{`` means
    JSONL, anything else is treated as CSV with a header row.
    """
    if fmt == "auto":
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        fmt = "jsonl" if first.startswith("{") else "csv"
    if fmt == "jsonl":
        records = tuple(_iter_jsonl(text, fields))
    elif fmt == "csv":
        records = tuple(_iter_csv(text, fields))
    else:
        raise MiscalError(f"unknown format {fmt!r}; expected 'jsonl', 'csv', or 'auto'")
    if not records:
        raise EmptyDatasetError("no records found in input")
    return Dataset(records=records)


def load_file(path: str, fmt: str = "auto", fields: FieldMap = FieldMap()) -> Dataset:
    """Read and parse a log file. ``fmt="auto"`` also honors the extension."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if fmt == "auto" and path.lower().endswith(".csv"):
        fmt = "csv"
    return parse_text(text, fmt=fmt, fields=fields)
