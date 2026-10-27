# UCI Adult Dataset: Raw Data Validation Report

## 1. Executive Summary

- **Total Rows:** 48,842
- **Total Columns:** 15
- **Overall Status:** PASSED (Valid Schema and Ranges)

## 2. Findings and Observations

- Column 'workclass' has 2799 missing values (5.73%): 2799 standard nulls and 0 explicit '?' markers.
- Column 'occupation' has 2809 missing values (5.75%): 2809 standard nulls and 0 explicit '?' markers.
- Column 'native-country' has 857 missing values (1.75%): 857 standard nulls and 0 explicit '?' markers.
- Detected 52 exact duplicate rows across all 15 columns.
- Detected 111 rows sharing identical feature vectors (excluding target).

## 3. Schema & Data Type Verification

| Column | Expected Type | Actual Dtype | Status | Non-Integer Count |
|---|---|---|---|---|
| `age` | integer | `int64` | **PASSED** | 0 |
| `workclass` | categorical | `category` | **PASSED** | 0 |
| `fnlwgt` | integer | `int64` | **PASSED** | 0 |
| `education` | categorical | `category` | **PASSED** | 0 |
| `education-num` | integer | `int64` | **PASSED** | 0 |
| `marital-status` | categorical | `category` | **PASSED** | 0 |
| `occupation` | categorical | `category` | **PASSED** | 0 |
| `relationship` | categorical | `category` | **PASSED** | 0 |
| `race` | categorical | `category` | **PASSED** | 0 |
| `sex` | categorical | `category` | **PASSED** | 0 |
| `capital-gain` | integer | `int64` | **PASSED** | 0 |
| `capital-loss` | integer | `int64` | **PASSED** | 0 |
| `hours-per-week` | integer | `int64` | **PASSED** | 0 |
| `native-country` | categorical | `category` | **PASSED** | 0 |
| `income` | categorical | `category` | **PASSED** | 0 |

## 4. Missing Value Analysis

> **Note on Missing Markers:** In the raw UCI Adult dataset, missing entries were traditionally encoded as `'?'`. When loaded via `fetch_openml(..., version=2)`, these are parsed into standard `NaN` values. Both explicit `'?'` markers and standard `NaN`s are tracked below.

| Column | Standard NaNs | Explicit `?` | Total Missing | % Missing |
|---|---|---|---|---|
| `age` | 0 | 0 | 0 | 0.0% |
| `workclass` | 2,799 | 0 | **2,799** | **5.731%** |
| `fnlwgt` | 0 | 0 | 0 | 0.0% |
| `education` | 0 | 0 | 0 | 0.0% |
| `education-num` | 0 | 0 | 0 | 0.0% |
| `marital-status` | 0 | 0 | 0 | 0.0% |
| `occupation` | 2,809 | 0 | **2,809** | **5.751%** |
| `relationship` | 0 | 0 | 0 | 0.0% |
| `race` | 0 | 0 | 0 | 0.0% |
| `sex` | 0 | 0 | 0 | 0.0% |
| `capital-gain` | 0 | 0 | 0 | 0.0% |
| `capital-loss` | 0 | 0 | 0 | 0.0% |
| `hours-per-week` | 0 | 0 | 0 | 0.0% |
| `native-country` | 857 | 0 | **857** | **1.755%** |
| `income` | 0 | 0 | 0 | 0.0% |

## 5. Numeric Range Validation

| Column | Expected Range | Actual Range | Low Violations | High Violations | Status |
|---|---|---|---|---|---|
| `age` | [17, 90] | [17.0, 90.0] | 0 | 0 | **PASSED** |
| `hours-per-week` | [1, 99] | [1.0, 99.0] | 0 | 0 | **PASSED** |
| `capital-gain` | [0, 99999] | [0.0, 99999.0] | 0 | 0 | **PASSED** |
| `capital-loss` | [0, 4356] | [0.0, 4356.0] | 0 | 0 | **PASSED** |

## 6. Duplicate Records

- **Exact Duplicate Rows (all 15 columns):** 52
- **Rows Sharing Duplicate Feature Vectors (excluding target):** 111

*Exact duplicate records in raw data are preserved in the raw ingestion layer and handled strictly according to the leakage-free protocol in the preprocessing step.*
