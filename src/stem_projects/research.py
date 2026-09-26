"""Analyze the 28 study records from my keratoconus extraction sheet."""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REQUIRED = ("study_id", "study_title", "year", "network_input_raw",
            "network_type_raw", "accuracy_raw", "sensitivity_raw", "specificity_raw",
            "extraction_note")
EXACT_PERCENT = re.compile(r"^(?:\d+(?:\.\d*)?|\.\d+)$")


def _input_type(description: str) -> str:
    text = description.lower().replace("topoagraphy", "topography")
    topography = any(term in text for term in ("topograph", "topographer", "videokeratograph"))
    tomography = "tomograph" in text
    if topography and tomography:
        return "both"
    if topography:
        return "topography"
    if tomography:
        return "tomography"
    return "unknown"


def _network_type(description: str) -> str:
    text = description.strip().upper()
    if not text:
        return "unknown"
    if "/" in text or "MULTIPLE" in text:
        return "multiple"
    if text.startswith("CNN"):
        return "cnn"
    if text == "RF":
        return "rf"
    if text == "SVM":
        return "svm"
    if text == "MLP":
        return "mlp"
    return "other"  # MLC, FFN, and unsupervised ML are not silently recoded.


def _accuracy(value: str) -> tuple[float, str]:
    text = value.strip()
    if not text or text in {"-", "—", "_"}:
        return np.nan, "missing"
    if text.startswith((">", "<", "≥", "≤")):
        return np.nan, "bound, not an exact value"
    if not EXACT_PERCENT.fullmatch(text):
        return np.nan, "non-numeric"
    number = float(text)
    if not 0 <= number <= 100:
        raise ValueError(f"Accuracy must be a percentage from 0 to 100: {text}")
    return number, "exact"


def load_studies(path: str | Path) -> pd.DataFrame:
    """Keep all original workbook cells; derive analysis columns without rewriting them."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if frame.empty or frame.study_id.eq("").any() or frame.study_id.duplicated().any():
        raise ValueError("Expected nonempty rows with unique study_id values")
    frame = frame.copy()
    frame["year"] = pd.to_numeric(frame.year, errors="raise")
    if ((frame.year % 1 != 0) | (frame.year < 1997) | (frame.year > 2022)).any():
        raise ValueError("Year must be an integer from 1997 to 2022")
    frame["input_type"] = frame.network_input_raw.map(_input_type)
    frame["network_type"] = frame.network_type_raw.map(_network_type)
    for metric in ("accuracy", "sensitivity", "specificity"):
        parsed = frame[f"{metric}_raw"].map(_accuracy)
        frame[metric] = parsed.map(lambda result: result[0])
        frame[f"{metric}_status"] = parsed.map(lambda result: result[1])
    frame.attrs["source_csv"] = str(path)
    return frame


def _spearman(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return {"n": len(x), "status": "insufficient observations or variation"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = stats.spearmanr(x, y)
    return {"n": len(x), "rho": float(result.statistic), "p_value": float(result.pvalue)}


def _kruskal(frame: pd.DataFrame, group: str) -> dict:
    included = frame[frame[group] != "unknown"]
    sizes = {str(key): int(value) for key, value in included.groupby(group).size().items()}
    groups = [part.accuracy.to_numpy() for _, part in included.groupby(group)]
    if len(groups) < 2 or len(np.unique(included.accuracy)) < 2:
        return {"n": len(included), "group_sizes": sizes,
                "status": "insufficient groups or variation"}
    result = stats.kruskal(*groups)
    return {"n": len(included), "group_sizes": sizes,
            "groups_with_fewer_than_two_studies": sorted(k for k, n in sizes.items() if n < 2),
            "h_statistic": float(result.statistic), "p_value": float(result.pvalue)}


def analyze(frame: pd.DataFrame) -> dict:
    exact = frame[frame.accuracy_status == "exact"]
    accuracy = exact.accuracy.to_numpy(dtype=float)
    if not len(exact):
        raise ValueError("No exact numeric study-level accuracy values were supplied")
    if 3 <= len(exact) <= 5000 and len(np.unique(accuracy)) >= 2:
        result = stats.shapiro(accuracy)
        normality = {"n": len(exact), "w_statistic": float(result.statistic),
                     "p_value": float(result.pvalue)}
    else:
        normality = {"n": len(exact), "status": "requires 3–5000 nonconstant observations"}

    metric_summaries = {}
    metric_availability = {}
    for metric in ("accuracy", "sensitivity", "specificity"):
        values = frame.loc[frame[f"{metric}_status"] == "exact", metric].to_numpy(dtype=float)
        metric_summaries[f"{metric}_percent_unweighted"] = ({
            "n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "minimum": float(np.min(values)), "maximum": float(np.max(values)),
        } if len(values) else {"n": 0, "status": "no exact values"})
        metric_availability[metric] = {
            "exact": int((frame[f"{metric}_status"] == "exact").sum()),
            "missing": int((frame[f"{metric}_status"] == "missing").sum()),
            "bound": int((frame[f"{metric}_status"] == "bound, not an exact value").sum()),
        }

    return {
        "source": frame.attrs.get("source_csv", "supplied study data"),
        "study_rows": int(len(frame)),
        "exact_accuracy_rows": int(len(exact)),
        "metric_availability": metric_availability,
        "source_notes": [
            {"study_id": row.study_id, "note": row.extraction_note}
            for row in frame.itertuples(index=False) if row.extraction_note.strip()
        ],
        "coding_review_flags": [
            {"study_id": row.study_id, "issue": "Study title mentions decision tree, but network_type_raw is CNN; verify against the article before interpreting the network comparison."}
            for row in frame.itertuples(index=False)
            if "decision tree" in row.study_title.lower()
            and row.network_type_raw.strip().upper() == "CNN"
        ],
        "excluded_accuracy_rows": [
            {"study_id": row.study_id, "accuracy_raw": row.accuracy_raw,
             "reason": row.accuracy_status}
            for row in frame.itertuples(index=False) if row.accuracy_status != "exact"
        ],
        **metric_summaries,
        "shapiro_wilk_accuracy": normality,
        "spearman_year_accuracy": _spearman(exact.year.to_numpy(), accuracy),
        "kruskal_wallis_network_type": _kruskal(exact, "network_type"),
        "kruskal_wallis_input_type": _kruskal(exact, "input_type"),
        "severity_accuracy": {
            "status": "not computed",
            "reason": "The sheet has severity-group counts but not a separate accuracy for each severity group. A study's overall accuracy cannot be assigned to each group.",
        },
        "limitations": (
            "Unweighted, exploratory study-level comparisons of the exact entries in my "
            "extraction sheet. Category labels were derived from the recorded descriptions; "
            "some source entries have notes or ambiguous network labels. Small categories, "
            "heterogeneous studies, and uncorrected multiple tests limit inference. "
            "These statistics may differ from the analyses in my paper."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path,
                        default=Path("keratoconus_analysis/data/kc_studies.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/research_analysis.json"))
    args = parser.parse_args()
    report = analyze(load_studies(args.csv))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
