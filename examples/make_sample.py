"""Regenerate the sample logs committed in this directory.

Simulates a support-ticket intent classifier whose verbalized confidences are
overconfident (the usual failure mode for LLM classifiers), plus a second run
after a hypothetical prompt fix that is much better calibrated. Everything is
seeded, so re-running this script reproduces the committed files byte for
byte:

    python examples/make_sample.py
"""

from __future__ import annotations

import json
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent

INTENTS = ("billing", "refund", "shipping", "account", "other")

TICKETS = (
    "I was charged twice this month",
    "where is my package",
    "please close my account",
    "the invoice amount looks wrong",
    "I want my money back",
    "how do I reset my password",
    "my order never arrived",
    "cancel my subscription",
    "the tracking number does not work",
    "can I change my delivery address",
    "why did my card get declined",
    "I returned the item two weeks ago",
)

# How the model verbalizes a numeric confidence, cycled deterministically.
STYLES = ("float", "percent", "fraction", "word", "float", "percent")

WORDS = (
    (0.95, "almost certain"),
    (0.90, "very likely"),
    (0.80, "confident"),
    (0.75, "likely"),
    (0.60, "medium"),
    (0.50, "maybe"),
    (0.30, "low"),
)


def _verbalize(conf: float, style: str) -> object:
    if style == "float":
        return round(conf, 2)
    if style == "percent":
        return f"{round(conf * 100)}%"
    if style == "fraction":
        return f"{round(conf * 10)}/10"
    # Nearest anchor word at or below the confidence.
    for anchor, word in WORDS:
        if conf >= anchor - 0.04:
            return word
    return WORDS[-1][1]


def _make_run(path: pathlib.Path, seed: int, n: int, honesty: float) -> None:
    """Write ``n`` records; ``honesty`` scales how truthful confidences are.

    The true probability of being correct is pulled toward 0.5 relative to the
    stated confidence: ``p_correct = 0.5 + honesty * (conf - 0.5)``. An honest
    model has honesty ~ 1.0; an overconfident one sits well below.
    """
    rng = random.Random(seed)
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            conf = rng.choice((0.55, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.98))
            p_correct = 0.5 + honesty * (conf - 0.5)
            correct = rng.random() < p_correct
            expected = rng.choice(INTENTS)
            predicted = expected if correct else rng.choice(
                tuple(x for x in INTENTS if x != expected)
            )
            record = {
                "id": f"ticket-{i + 1:04d}",
                "input": rng.choice(TICKETS),
                "predicted": predicted,
                "expected": expected,
                "confidence": _verbalize(conf, STYLES[i % len(STYLES)]),
            }
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    plural = "record" if n == 1 else "records"
    print(f"wrote {path.name} ({n} {plural}, honesty={honesty})")


if __name__ == "__main__":
    _make_run(HERE / "sample_run.jsonl", seed=20260712, n=200, honesty=0.35)
    _make_run(HERE / "sample_run_v2.jsonl", seed=20260713, n=200, honesty=0.85)
