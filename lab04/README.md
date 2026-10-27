# Machine Learning Lab 4: Data Engineering, Validation, Leakage-Free Preprocessing, and Model Auditing

This folder contains the complete, reproducible implementation for **Lab 4: Data Engineering, Validation, Leakage-Free Preprocessing, Feature Engineering, and Model Auditing** using the **UCI Adult (Census Income)** dataset.

---

## 1. Project Directory Structure

```text
lab04/
├── README.md                      # Setup and execution guide
├── report.md                      # Comprehensive technical report and empirical findings
├── data_card.md                   # Formal data card (sources, schema, biases, limitations)
├── CLASS_PREPARATION.md           # Student presentation and oral exam defense guide
├── config.yaml                    # Centralized parameters, seeds, ranges, and paths
├── requirements.txt               # Exact dependencies and library versions
├── run_pipeline.py                # Single-command executable entry point
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # Raw ingestion from OpenML v2 and dimension checks
│   ├── validation.py              # Schema, missingness, numeric ranges, and duplicate audit
│   ├── preprocessing.py           # 70/30 stratified split, train dedup, clipper, imputer, joblib
│   ├── feature_engineering.py     # Safe derivation of capital_net and hours_per_age
│   ├── modeling.py                # Logistic Regression baseline comparison and metrics
│   ├── label_audit.py             # 50-row reproducible sample and conflicting label audit
│   └── reproducibility.py         # Deterministic SHA256 hashing and manifest generation
├── tests/
│   ├── conftest.py
│   └── test_pipeline.py           # 9 automated unit and integration tests
└── results/                       # Persisted experiment artifacts
    ├── validation_report.json     # Machine-readable validation findings
    ├── validation_report.md       # Human-readable validation report
    ├── model_comparison.csv       # Baseline vs Feature Engineered metrics table
    ├── model_comparison.md       # Formatted Markdown model comparison
    ├── preprocessor.joblib        # Fitted scikit-learn preprocessor
    ├── sampled_50_rows.csv        # 50 reproducibly sampled records (seed 42)
    ├── label_audit_checklist.md   # Structured checklist for manual review
    ├── label_audit_conflicts.csv  # 10 records with identical features & conflicting labels
    └── reproducibility_manifest.json # Complete hashes, shapes, seeds, and library versions
```

---

## 2. Quick Start: One-Command Execution

To execute the entire pipeline end-to-end (ingestion, validation, preprocessing, feature engineering, modeling, auditing, and hashing):

```bash
# 1. Activate project virtual environment
source .venv/bin/activate

# 2. Run the complete pipeline
python lab04/run_pipeline.py
```

### Optional Arguments:
- `--config <path>`: Use a custom YAML configuration file.
- `--force-download`: Force re-fetching the dataset from OpenML v2 even if local raw caches exist.

---

## 3. Running Automated Tests

Run the test suite using `pytest`:

```bash
pytest lab04/tests/ -v
```

All 9 automated tests verify:
1. `test_raw_data_dimensions`: Exact row count (48,842) and column count (15).
2. `test_validation_schema_and_ranges`: Range and schema compliance with 0 violations.
3. `test_stratified_split_and_no_leakage`: 70/30 proportions, target stratification, train-only deduplication.
4. `test_target_leakage_prevention`: Complete removal of target from input matrices.
5. `test_numeric_outlier_clipper`: Proper quantile fitting on train data and application to test data.
6. `test_feature_engineering_safety`: Safe division policy and zero/negative denominator handling.
7. `test_preprocessor_serialization`: Clean `joblib` saving and loading (supports string and Path objects) without numeric divergence.
8. `test_base_dir_string_normalization`: Robust handling of `base_dir` as either string or `Path` object across modules.
9. `test_label_quality_audit`: 50-row sampling integrity and conflicting-label detection (5 pairs / 10 rows).

---

## 4. Key Experimental Findings

| Metric | Baseline (14 Raw Features) | With Engineered Features (+2) | Difference |
|---|---|---|---|
| **Accuracy** | 0.8540 (85.40%) | **0.8578 (85.78%)** | **+0.0038 (+0.38%)** |
| **F1-Score** | 0.6601 | **0.6696** | **+0.0095 (+0.95%)** |
| **Precision** | 0.7448 | **0.7537** | **+0.0089** |
| **Recall** | 0.5927 | **0.6024** | **+0.0097** |
| **ROC-AUC** | 0.9058 | **0.9096** | **+0.0038** |

- **Derived Features:**
  - `capital_net = capital-gain - capital-loss`
  - `hours_per_age = hours-per-week / (age - 16)`
- **Conflicting Labels Identified:** 10 rows (5 pairs) where all 14 demographic attributes are identical but one record earns `<=50K` and the other earns `>50K`.
- **Training Deduplication Shift:** 24 exact duplicates dropped from training split only; test set remained untouched (14,653 rows); target class shift was limited to $\pm 0.0080\%$.
