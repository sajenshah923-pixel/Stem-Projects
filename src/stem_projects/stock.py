"""Leakage-aware preparation for the stock movement classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = ("down", "flat", "up")


def load_closes(path: str, close_column: str = "Close", date_column: str = "Date") -> np.ndarray:
    frame = pd.read_csv(path)
    if close_column not in frame:
        raise ValueError(f"Missing price column {close_column!r}")
    if date_column in frame:
        dates = pd.to_datetime(frame[date_column], errors="raise")
        if dates.isna().any() or dates.duplicated().any():
            raise ValueError("Dates must be present and unique")
        frame = frame.assign(_date=dates).sort_values("_date")
    prices = pd.to_numeric(frame[close_column], errors="raise").to_numpy(dtype=float)
    if len(prices) < 5 or not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("At least five finite, positive closing prices are required")
    return prices


def make_examples(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """At time t use log(P[t]/P[t-1]); label sign(P[t+1]-P[t])."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 5 or np.any(prices <= 0) or not np.all(np.isfinite(prices)):
        raise ValueError("At least five finite, positive prices are required")
    returns = np.diff(np.log(prices))
    x = returns[:-1].reshape(-1, 1)
    direction = np.sign(np.diff(prices)[1:]).astype(int)
    y = direction + 1  # down=0, flat=1, up=2
    return x, y


def chronological_split(x: np.ndarray, y: np.ndarray, train_fraction: float = 0.8):
    if not 0 < train_fraction < 1 or len(x) != len(y) or len(x) < 4:
        raise ValueError("Need matching arrays, at least four examples, and a fraction in (0, 1)")
    boundary = max(2, min(len(x) - 1, int(len(x) * train_fraction)))
    return x[:boundary], x[boundary:], y[:boundary], y[boundary:]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if not len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("Predictions must match a nonempty set of labels")
    if np.any((y_true < 0) | (y_true > 2) | (y_pred < 0) | (y_pred > 2)):
        raise ValueError("Labels must be 0, 1, or 2")
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    return {"accuracy": float(np.mean(y_true == y_pred)), "labels": LABELS,
            "confusion_matrix_rows_actual": matrix.tolist(), "test_examples": len(y_true)}

