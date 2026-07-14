# miscal record format

miscal consumes one *record* per classifier decision, from a JSONL file (one
JSON object per line) or a CSV file with a header row. This document is the
authoritative description of how records are interpreted; the same rules are
enforced by `tests/test_records.py` and `tests/test_verbal.py`.

## Required information

Every record must yield two values:

1. **confidence** — how sure the model said it was, in any supported form.
2. **correctness** — whether the decision was right, either logged directly
   or derived from a predicted/expected label pair.

Anything else in the record is preserved as-is (and written back out by
`miscal fit --apply`) but ignored by the metrics.

## Field resolution

Fields are found by trying alias lists in order. An explicit CLI flag
(`--confidence-field`, `--correct-field`, `--predicted-field`,
`--expected-field`) overrides every alias, and it is an error if the named
field is missing from a record.

| Value | Aliases (tried in order) |
|---|---|
| confidence | `confidence`, `conf`, `p`, `prob`, `probability`, `score` |
| correctness | `correct`, `is_correct`, `hit`, `success` |
| predicted label | `predicted`, `prediction`, `pred`, `output` |
| expected label | `expected`, `gold`, `label`, `truth`, `answer` |

If no correctness field is present, miscal compares the predicted label
against the expected one — trimmed and case-insensitively, because `"Spam "`
vs `"spam"` is an LLM formatting quirk, not a wrong answer. A record with
neither a correctness field nor a complete label pair is an error, reported
with its line number.

## Confidence forms

| Input | Interpreted as | Rule |
|---|---|---|
| `0.85` | 0.85 | numbers in `[0, 1]` are probabilities |
| `85` / `99.5` | 0.85 / 0.995 | numbers in `(1, 100]` are percentages |
| `-0.2` / `150` | error | out-of-range values are rejected, never clamped |
| `"85%"` / `"85 %"` | 0.85 | percentage strings, `[0%, 100%]` |
| `"9/10"` | 0.9 | fractions; must land in `[0, 1]`, denominator ≠ 0 |
| `"0.85"` / `"85"` | 0.85 | numeric strings follow the number rules |
| `"very likely"` | 0.90 | anchor words, case-insensitive (full table in the README) |
| `true`, `NaN`, `""` | error | booleans, NaN, and empty strings are rejected |

The bare-number percent rule exists because prompts asking for "confidence
from 0 to 100" are common; `1` is deliberately read as certainty (`p = 1.0`),
not 1%, since `[0, 1]` takes precedence.

## Correctness forms

Booleans, the integers `0`/`1`, and the strings `true/t/yes/y/1` and
`false/f/no/n/0` (case-insensitive) are accepted. Anything else — including
"partially correct" values — is an error, because a silent coercion would
corrupt every downstream metric.

## Errors

All parse errors name the 1-based line of the offending record (CSV line
numbers count the header as line 1). An input with zero records is an error
rather than an empty report.
