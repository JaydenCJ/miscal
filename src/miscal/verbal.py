"""Turn what an LLM *said* about its confidence into a probability.

Production logs rarely contain a clean float. Models verbalize confidence as
``"85%"``, ``"9/10"``, ``"very likely"``, or a bare number on an unstated
scale. This module maps every form miscal accepts onto a probability in
``[0.0, 1.0]`` with one deterministic rule set, documented in the README so
that a report is always reproducible from the raw log.

The word scale follows the spirit of the "words of estimative probability"
literature: fixed anchor points, not model-specific tuning. If your prompt
uses a different vocabulary, log numeric confidences instead — parsing prose
is a convenience, not a substitute for structured logging.
"""

from __future__ import annotations

import re

from .errors import ConfidenceError

# Anchor probabilities for verbal confidence expressions. The whole trimmed,
# lowercased string must equal a key exactly — there is no substring matching,
# so "very unlikely" (0.08) can never be misread as "unlikely" (0.25).
WORD_SCALE: dict[str, float] = {
    "almost certain": 0.97,
    "almost certainly": 0.97,
    "highly likely": 0.92,
    "very likely": 0.90,
    "very confident": 0.90,
    "highly confident": 0.92,
    "very unlikely": 0.08,
    "highly unlikely": 0.05,
    "almost impossible": 0.03,
    "fairly confident": 0.75,
    "fairly likely": 0.70,
    "certain": 0.97,
    "certainly": 0.97,
    "definitely": 0.95,
    "confident": 0.80,
    "likely": 0.75,
    "probable": 0.75,
    "probably": 0.70,
    "possibly": 0.50,
    "possible": 0.50,
    "maybe": 0.50,
    "perhaps": 0.50,
    "uncertain": 0.50,
    "unsure": 0.50,
    "toss-up": 0.50,
    "doubtful": 0.20,
    "unlikely": 0.25,
    "improbable": 0.20,
    "impossible": 0.03,
    "high": 0.90,
    "medium": 0.60,
    "moderate": 0.60,
    "low": 0.30,
}

_PERCENT_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*%$")
_FRACTION_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)$")
_NUMBER_RE = re.compile(r"^-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?$")


def _from_number(value: float) -> float:
    """Interpret a bare number as a probability.

    Rule: values in ``[0, 1]`` are probabilities as-is; values in ``(1, 100]``
    are percentages (LLMs asked for "confidence 0-100" answer on that scale).
    Anything negative or above 100 is rejected rather than clamped, because it
    signals a logging bug the user should see, not smooth over.
    """
    if 0.0 <= value <= 1.0:
        return value
    if 1.0 < value <= 100.0:
        return value / 100.0
    raise ConfidenceError(
        f"confidence {value!r} is outside [0, 1] and [0, 100]; refusing to guess a scale"
    )


def parse_confidence(raw: object) -> float:
    """Parse any supported confidence representation into a probability.

    Accepts:

    * ``float``/``int`` — ``[0, 1]`` as-is, ``(1, 100]`` treated as percent.
    * ``"85%"`` / ``"85 %"`` — percentage strings.
    * ``"9/10"`` — fractions (numerator/denominator).
    * ``"0.85"`` / ``"85"`` — numeric strings, same scale rule as numbers.
    * ``"very likely"`` — anchor words from :data:`WORD_SCALE`.

    Raises :class:`ConfidenceError` for anything else. Never returns NaN.
    """
    if isinstance(raw, bool):
        raise ConfidenceError("confidence must be a number or string, got a boolean")
    if isinstance(raw, (int, float)):
        if raw != raw:  # NaN check without math.isnan on non-floats
            raise ConfidenceError("confidence is NaN")
        return _from_number(float(raw))
    if isinstance(raw, str):
        text = raw.strip().strip(".,;:!").strip().lower()
        if not text:
            raise ConfidenceError("confidence string is empty")
        m = _PERCENT_RE.match(text)
        if m:
            return _percent(m)
        m = _FRACTION_RE.match(text)
        if m:
            return _fraction(m)
        if _NUMBER_RE.match(text):
            return _from_number(float(text))
        if text in WORD_SCALE:
            return WORD_SCALE[text]
        raise ConfidenceError(
            f"cannot interpret confidence {raw!r}; expected a probability, percentage, "
            f"fraction, or one of {len(WORD_SCALE)} anchor words (see README)"
        )
    raise ConfidenceError(f"confidence must be a number or string, got {type(raw).__name__}")


def _percent(m: re.Match) -> float:
    value = float(m.group(1))
    if not 0.0 <= value <= 100.0:
        raise ConfidenceError(f"percentage {value}% is outside [0%, 100%]")
    return value / 100.0


def _fraction(m: re.Match) -> float:
    numerator = float(m.group(1))
    denominator = float(m.group(2))
    if denominator == 0:
        raise ConfidenceError("fraction confidence has a zero denominator")
    value = numerator / denominator
    if not 0.0 <= value <= 1.0:
        raise ConfidenceError(f"fraction {m.group(0)!r} evaluates to {value}, outside [0, 1]")
    return value
