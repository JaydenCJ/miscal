"""Shared fixtures and dataset factories for the miscal test suite.

Everything here is deterministic: synthetic datasets are built from explicit
formulas or a seeded ``random.Random``, never from wall-clock time or global
random state, so every test run sees identical numbers.
"""

from __future__ import annotations

import json
import pathlib
import random

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture()
def sample_run_path() -> pathlib.Path:
    """The committed example log used by the README quickstart."""
    return REPO_ROOT / "examples" / "sample_run.jsonl"


@pytest.fixture()
def sample_run_v2_path() -> pathlib.Path:
    return REPO_ROOT / "examples" / "sample_run_v2.jsonl"


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    """Write dict rows as a JSONL file and return the path."""
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def hand_dataset() -> tuple[list[float], list[int]]:
    """The hand-computed reference dataset used across metric tests.

    With 10 equal-width bins: bin [0.9, 1.0] has conf 0.9, acc 0.5 (gap 0.4,
    n=4); bin [0.6, 0.7) has conf 0.6, acc 0.5 (gap 0.1, n=2); bin [0.3, 0.4)
    has conf 0.3, acc 0.0 (gap 0.3, n=1). Therefore:

    * ECE   = (4*0.4 + 2*0.1 + 1*0.3) / 7 = 2.1 / 7 = 0.3
    * MCE   = 0.4
    * Brier = (2*0.01 + 2*0.81 + 0.16 + 0.36 + 0.09) / 7 = 2.25 / 7
    """
    confidences = [0.9, 0.9, 0.9, 0.9, 0.6, 0.6, 0.3]
    outcomes = [1, 1, 0, 0, 1, 0, 0]
    return confidences, outcomes


def synthetic_run(
    n: int, honesty: float, seed: int = 7
) -> tuple[list[float], list[int]]:
    """A seeded synthetic classifier run for temperature-fitting tests.

    ``honesty`` pulls the true correctness probability toward the stated
    confidence: 1.0 means perfectly calibrated in expectation, values below
    1.0 mean overconfident, above 1.0 mean underconfident.
    """
    rng = random.Random(seed)
    confidences: list[float] = []
    outcomes: list[int] = []
    levels = (0.55, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
    for _ in range(n):
        conf = rng.choice(levels)
        p_correct = min(max(0.5 + honesty * (conf - 0.5), 0.02), 0.98)
        confidences.append(conf)
        outcomes.append(1 if rng.random() < p_correct else 0)
    return confidences, outcomes
