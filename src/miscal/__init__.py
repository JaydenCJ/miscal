"""miscal — calibration reports for LLM classifiers.

Zero-dependency calibration math for logged confidences: ECE, Brier score
and its decomposition, log loss, reliability diagrams as standalone SVG,
temperature scaling, and a CLI that gates CI on calibration regressions.
"""

from .binning import Bin, equal_mass_bins, equal_width_bins, make_bins
from .errors import ConfidenceError, EmptyDatasetError, MiscalError, RecordError
from .metrics import (
    Metrics,
    accuracy,
    brier_decomposition,
    brier_score,
    confidence_gap,
    ece,
    evaluate,
    expected_calibration_error,
    log_loss,
    mce,
    mean_confidence,
)
from .records import Dataset, FieldMap, Record, load_file, parse_text, record_from_dict
from .report import compare_report, json_report, text_report
from .svg import render_reliability_diagram
from .temperature import TemperatureFit, apply_temperature, fit_temperature
from .verbal import WORD_SCALE, parse_confidence

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Bin",
    "ConfidenceError",
    "Dataset",
    "EmptyDatasetError",
    "FieldMap",
    "Metrics",
    "MiscalError",
    "Record",
    "RecordError",
    "TemperatureFit",
    "WORD_SCALE",
    "accuracy",
    "apply_temperature",
    "brier_decomposition",
    "brier_score",
    "compare_report",
    "confidence_gap",
    "ece",
    "equal_mass_bins",
    "equal_width_bins",
    "evaluate",
    "expected_calibration_error",
    "fit_temperature",
    "json_report",
    "load_file",
    "log_loss",
    "make_bins",
    "mce",
    "mean_confidence",
    "parse_confidence",
    "parse_text",
    "record_from_dict",
    "render_reliability_diagram",
    "text_report",
]
