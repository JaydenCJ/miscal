"""Exception hierarchy for miscal.

Everything raised deliberately by the library derives from :class:`MiscalError`
so callers (and the CLI) can catch one type and print a clean message instead
of a traceback.
"""

from __future__ import annotations


class MiscalError(Exception):
    """Base class for all errors raised by miscal."""


class RecordError(MiscalError):
    """A logged record could not be parsed.

    Carries the 1-based line number of the offending record when known, so CLI
    error messages point users at the exact line of their log file.
    """

    def __init__(self, message: str, line: int | None = None):
        self.line = line
        if line is not None:
            message = f"line {line}: {message}"
        super().__init__(message)


class ConfidenceError(MiscalError):
    """A confidence value could not be interpreted as a probability."""


class EmptyDatasetError(MiscalError):
    """An operation that needs at least one record received none."""
