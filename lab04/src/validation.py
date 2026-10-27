"""

Data Validation Module for UCI Adult Dataset.



Performs schema verification, data type checking, missing-value detection (including '?'),

numeric range validation, and duplicate detection without dropping rows.

Generates human-readable Markdown and machine-readable JSON validation reports.

"""



import json

import logging

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple



import numpy as np

import pandas as pd



logger = logging.getLogger(__name__)





def validate_raw_data(

    df: pd.DataFrame,

    config: Dict[str, Any],

    base_dir: Optional[Path] = None

) -> Tuple[Dict[str, Any], bool]:

    """

    Validate the raw DataFrame against expected schema, ranges, missingness, and duplicates.



    Args:

        df: Input DataFrame (with standardized target column name 'income').

        config: Configuration dictionary.

        base_dir: Base directory for output report files.



    Returns:

        Tuple of (validation_results_dict, is_valid_boolean).

    """

    if base_dir is None:

        base_dir = Path(__file__).resolve().parent.parent

    else:

        base_dir = Path(base_dir)



    validation_cfg = config.get("validation", {})

    expected_schema = validation_cfg.get("expected_schema", {})

    numeric_ranges = validation_cfg.get("numeric_ranges", {})

    missing_markers = validation_cfg.get("missing_markers", ["?", "nan", "NaN", "None"])



    total_rows, total_cols = df.shape

    report: Dict[str, Any] = {

        "summary": {

            "total_rows": total_rows,

            "total_columns": total_cols,

            "all_passed": True

        },

        "schema_validation": {},

        "missing_values": {},

        "range_validation": {},

        "duplicates": {},

        "findings": []

    }



    # 1. Schema Validation (Column existence and types)

    missing_cols = [c for c in expected_schema.keys() if c not in df.columns]

    unexpected_cols = [c for c in df.columns if c not in expected_schema.keys()]



    report["schema_validation"]["missing_columns"] = missing_cols

    report["schema_validation"]["unexpected_columns"] = unexpected_cols



    if missing_cols:

        report["summary"]["all_passed"] = False

        report["findings"].append(f"Missing expected columns: {missing_cols}")



    type_checks = {}

    for col, expected_type in expected_schema.items():

        if col not in df.columns:

            continue

        col_series = df[col]

        actual_dtype = str(col_series.dtype)



        if expected_type == "integer":

            # Allowed: integer or float (if containing NaNs, though Adult numeric cols have 0 NaNs)

            is_valid_type = pd.api.types.is_numeric_dtype(col_series)

            # Check for non-integer numeric values

            non_int_count = 0

            if is_valid_type:

                non_null = col_series.dropna()

                non_int_count = int((non_null != non_null.astype(int)).sum())

            valid_col = is_valid_type and (non_int_count == 0)

        elif expected_type == "categorical":

            valid_col = (

                isinstance(col_series.dtype, pd.CategoricalDtype)

                or pd.api.types.is_object_dtype(col_series)

                or pd.api.types.is_string_dtype(col_series)

            )

            non_int_count = 0

        else:

            valid_col = True

            non_int_count = 0



        type_checks[col] = {

            "expected_type": expected_type,

            "actual_dtype": actual_dtype,

            "type_valid": bool(valid_col),

            "non_integer_count": non_int_count

        }

        if not valid_col:

            report["summary"]["all_passed"] = False

            report["findings"].append(

                f"Column '{col}' failed type check: expected {expected_type}, got {actual_dtype}"

            )



    report["schema_validation"]["type_checks"] = type_checks



    # 2. Missing Value Analysis (distinguish nulls and explicit markers like '?')

    missing_stats = {}

    for col in df.columns:

        series = df[col]

        standard_nulls = int(series.isna().sum())



        # Check for literal string '?' or other markers in object/category representations

        q_count = 0

        if isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series):

            # check category strings or string series

            str_series = series.astype(str).str.strip()

            q_count = int(str_series.isin(["?", "None", "nan", "NaN"]).sum())

            # Don't double count if already null

            q_count = max(0, q_count - standard_nulls)



        total_missing = standard_nulls + q_count

        pct_missing = float((total_missing / total_rows) * 100)



        missing_stats[col] = {

            "standard_null_count": standard_nulls,

            "question_mark_count": q_count,

            "total_missing_count": total_missing,

            "pct_missing": round(pct_missing, 3)

        }



        if total_missing > 0:

            report["findings"].append(

                f"Column '{col}' has {total_missing} missing values ({pct_missing:.2f}%): "

                f"{standard_nulls} standard nulls and {q_count} explicit '?' markers."

            )



    report["missing_values"] = missing_stats



    # 3. Numeric Range Validation

    range_results = {}

    for col, bounds in numeric_ranges.items():

        if col not in df.columns:

            continue

        min_expected = bounds["min"]

        max_expected = bounds["max"]

        series = df[col].dropna()



        actual_min = float(series.min()) if not series.empty else None

        actual_max = float(series.max()) if not series.empty else None



        violations_low = int((series < min_expected).sum())

        violations_high = int((series > max_expected).sum())

        total_violations = violations_low + violations_high



        range_results[col] = {

            "expected_range": [min_expected, max_expected],

            "actual_range": [actual_min, actual_max],

            "violations_low": violations_low,

            "violations_high": violations_high,

            "total_violations": total_violations,

            "range_valid": bool(total_violations == 0)

        }



        if total_violations > 0:

            report["summary"]["all_passed"] = False

            report["findings"].append(

                f"Column '{col}' had {total_violations} range violations (outside [{min_expected}, {max_expected}])."

            )



    report["range_validation"] = range_results



    # 4. Duplicate Records Check

    exact_duplicates = int(df.duplicated().sum())

    features = [c for c in df.columns if c != "income"]

    feature_duplicates = int(df.duplicated(subset=features, keep=False).sum())



    report["duplicates"] = {

        "exact_duplicate_rows": exact_duplicates,

        "rows_sharing_features": feature_duplicates

    }

    report["findings"].append(

        f"Detected {exact_duplicates} exact duplicate rows across all 15 columns."

    )

    report["findings"].append(

        f"Detected {feature_duplicates} rows sharing identical feature vectors (excluding target)."

    )



    # 5. Save Reports

    results_cfg = config.get("results", {})

    json_path = base_dir / results_cfg.get("validation_report_json", "results/validation_report.json")

    md_path = base_dir / results_cfg.get("validation_report_md", "results/validation_report.md")



    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:

        json.dump(report, f, indent=2)



    # Markdown report generation

    _write_markdown_report(report, md_path)



    logger.info(f"Validation finished. Passed: {report['summary']['all_passed']}")

    logger.info(f"Saved validation reports to {json_path} and {md_path}")



    return report, report["summary"]["all_passed"]





def _write_markdown_report(report: Dict[str, Any], output_path: Path) -> None:

    """Write human-readable Markdown validation report."""

    md = []

    md.append("# UCI Adult Dataset: Raw Data Validation Report\n")

    md.append("## 1. Executive Summary\n")

    md.append(f"- **Total Rows:** {report['summary']['total_rows']:,}")

    md.append(f"- **Total Columns:** {report['summary']['total_columns']}")

    md.append(f"- **Overall Status:** {'PASSED (Valid Schema and Ranges)' if report['summary']['all_passed'] else 'ATTENTION NEEDED'}\n")



    md.append("## 2. Findings and Observations\n")

    for f in report["findings"]:

        md.append(f"- {f}")

    md.append("")



    md.append("## 3. Schema & Data Type Verification\n")

    md.append("| Column | Expected Type | Actual Dtype | Status | Non-Integer Count |")

    md.append("|---|---|---|---|---|")

    for col, info in report["schema_validation"]["type_checks"].items():

        status = "PASSED" if info["type_valid"] else "FAILED"

        md.append(f"| `{col}` | {info['expected_type']} | `{info['actual_dtype']}` | **{status}** | {info['non_integer_count']} |")

    md.append("")



    md.append("## 4. Missing Value Analysis\n")

    md.append("> **Note on Missing Markers:** In the raw UCI Adult dataset, missing entries were traditionally encoded as `'?'`. When loaded via `fetch_openml(..., version=2)`, these are parsed into standard `NaN` values. Both explicit `'?'` markers and standard `NaN`s are tracked below.")

    md.append("")

    md.append("| Column | Standard NaNs | Explicit `?` | Total Missing | % Missing |")

    md.append("|---|---|---|---|---|")

    for col, info in report["missing_values"].items():

        if info["total_missing_count"] > 0:

            md.append(f"| `{col}` | {info['standard_null_count']:,} | {info['question_mark_count']} | **{info['total_missing_count']:,}** | **{info['pct_missing']}%** |")

        else:

            md.append(f"| `{col}` | 0 | 0 | 0 | 0.0% |")

    md.append("")



    md.append("## 5. Numeric Range Validation\n")

    md.append("| Column | Expected Range | Actual Range | Low Violations | High Violations | Status |")

    md.append("|---|---|---|---|---|---|")

    for col, info in report["range_validation"].items():

        status = "PASSED" if info["range_valid"] else "VIOLATION"

        exp = f"[{info['expected_range'][0]}, {info['expected_range'][1]}]"

        act = f"[{info['actual_range'][0]}, {info['actual_range'][1]}]"

        md.append(f"| `{col}` | {exp} | {act} | {info['violations_low']} | {info['violations_high']} | **{status}** |")

    md.append("")



    md.append("## 6. Duplicate Records\n")

    md.append(f"- **Exact Duplicate Rows (all 15 columns):** {report['duplicates']['exact_duplicate_rows']:,}")

    md.append(f"- **Rows Sharing Duplicate Feature Vectors (excluding target):** {report['duplicates']['rows_sharing_features']:,}")

    md.append("\n*Exact duplicate records in raw data are preserved in the raw ingestion layer and handled strictly according to the leakage-free protocol in the preprocessing step.*")



    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:

        f.write("\n".join(md) + "\n")
