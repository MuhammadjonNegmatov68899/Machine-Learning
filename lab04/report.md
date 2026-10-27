# Machine Learning Lab 4: Data Engineering, Leakage-Free Preprocessing, and Model Auditing

## 1. Executive Summary

This report documents the rigorous engineering and empirical evaluation of the **UCI Adult (Census Income)** dataset for **Lab 4**. The primary objective is to build a reproducible, leakage-free machine learning workflow that predicts whether an individual earns more than \$50,000 annually (`<=50K` vs `>50K`), while adhering to strict data validation, ethical auditing, and preprocessing isolation standards.

### Key Measured Outcomes
- **Raw Data Ingestion:** 48,842 rows and 15 columns verified against OpenML v2 specifications.
- **Data Validation:** 100% of numeric columns fall strictly within domain-valid bounds; exactly 52 duplicate records detected in the full raw dataset.
- **Leakage-Free Splitting:** Stratified 70/30 split produced 34,189 training rows and 14,653 test rows.
- **Deduplication Audit:** 24 exact duplicate rows were dropped from the training split only (34,189 $\rightarrow$ 34,165 rows). The test set remained completely untouched (14,653 rows). The class distribution shifted by merely $\pm 0.0080\%$.
- **Model Comparison:**
  - **Baseline Logistic Regression:** Accuracy = **85.40%**, F1-Score = **0.6601**, ROC-AUC = **0.9058**.
  - **With Derived Features (`capital_net`, `hours_per_age`):** Accuracy = **85.78%**, F1-Score = **0.6696**, ROC-AUC = **0.9096**.
  - **Empirical Impact:** Both Accuracy (+0.38 percentage points) and F1-Score (+0.0095 points / ~1 percentage point) genuinely improved on the untouched test partition.
- **Label-Quality Audit:** Identified exactly 10 rows (5 unique feature vector pairs) where all 14 input attributes are 100% identical but their income labels contradict each other (`<=50K` vs `>50K`).

---

## 2. Ingestion and Raw Data Integrity

The ingestion pipeline (`src/dataset.py`) retrieves `adult` version 2 from OpenML via `fetch_openml(name='adult', version=2, as_frame=True)`.

### Dimension Verification
| Dimension | Expected | Measured | Status |
|---|---|---|---|
| Rows | 48,842 | 48,842 | **PASSED** |
| Columns | 15 | 15 | **PASSED** |

### Data Separation Strategy
Raw data is persisted immutably to `data/raw/adult_raw.parquet` and `data/raw/adult_raw.csv`. The target column in OpenML (`class`) is mapped to `income` for clarity, while raw files retain original column headers.

---

## 3. Data Validation & Schema Enforcement

The validation module (`src/validation.py`) enforces strict type and range constraints without silently discarding records:

### Schema and Type Verification
All 15 columns were verified against expected schema types (6 integer features, 8 categorical features, 1 categorical target). No non-integer values were detected in integer fields.

### Missing Value Analysis
Missing markers in the raw Adult census were originally `'?'`. OpenML converts these to standard `NaN` values:
- `workclass`: 2,799 missing values (5.73%)
- `occupation`: 2,809 missing values (5.75%)
- `native-country`: 857 missing values (1.75%)
- Remaining 11 features and `income`: 0 missing values.

### Numeric Range Bounds
| Column | Expected Bound | Actual Bound | Violations | Status |
|---|---|---|---|---|
| `age` | [17, 90] | [17, 90] | 0 | **PASSED** |
| `hours-per-week` | [1, 99] | [1, 99] | 0 | **PASSED** |
| `capital-gain` | [0, 99999] | [0, 99999] | 0 | **PASSED** |
| `capital-loss` | [0, 4356] | [0, 4356] | 0 | **PASSED** |

---

## 4. Leakage-Free Preprocessing Protocol

Data leakage occurs whenever information from the evaluation or unseen real-world environment contaminates the model training phase. In this implementation, leakage is systematically prevented through five architectural constraints:

1. **Stratification Before Transformation:**
   - The dataset is split into 70% train (34,189 rows) and 30% test (14,653 rows) using `stratify=y` with fixed seed 42.
   - Target ratios:
     - Full dataset: 76.0718% `<=50K`, 23.9282% `>50K`
     - Train split: 76.0713% `<=50K`, 23.9287% `>50K`
     - Test split: 76.0732% `<=50K`, 23.9268% `>50K`
2. **Training-Only Deduplication:**
   - Exact duplicates are removed **strictly from the training set**.
   - Exactly 24 duplicates were removed from the training split (34,189 $\rightarrow$ 34,165).
   - The test set was **not deduplicated or cleaned**, preserving its realistic operational distribution.
   - **Distribution Shift Analysis:** Dropping 24 training duplicates shifted the training target proportion by only $\pm 0.0080\%$ (76.071% $\rightarrow$ 76.063% `<=50K`), demonstrating that deduplication did not perturb the training prior.
3. **Training-Fitted Imputation:**
   - Categorical missing values are imputed using the **training set mode** (`most_frequent`).
   - Numeric missing values are imputed using the **training set median**.
   - Test data is transformed using the imputers fitted on training data; test statistics are never computed.
4. **Training-Fitted Outlier Clipping (`NumericOutlierClipper`):**
   - Outliers are clipped between the 1st and 99th percentiles computed solely on `X_train`.
   - The learned percentile thresholds (`bounds_`) are persisted and applied to `X_test`.
5. **Absolute Target Exclusion:**
   - The target variable `income` is severed from features before preprocessor construction, eliminating target leakage.

---

## 5. Feature Engineering & Empirical Baseline Comparison

Two domain features were constructed:
1. `capital_net = capital-gain - capital-loss`
   - Expresses net financial yield directly to linear decision surfaces.
2. `hours_per_age = hours-per-week / (age - 16)`
   - Quantifies working intensity adjusted for career seniority (years since minimum legal working age of 16).
   - **Safe Division Policy:** If `age - 16 <= 0` or is missing, the denominator is defensively clamped to `1.0`.

### Empirical Results Comparison

Both models were trained using identical Logistic Regression settings (`max_iter=1000`, `solver='lbfgs'`, `random_state=42`) on independently fitted preprocessors:

| Model | Features | Accuracy | F1-Score | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|---|
| **Baseline (Without Derived Features)** | 14 | 0.8540 | 0.6601 | 0.7448 | 0.5927 | 0.9058 |
| **With Derived Features** | 16 | **0.8578** | **0.6696** | **0.7537** | **0.6024** | **0.9096** |
| **Absolute Improvement** | **+2** | **+0.0038** | **+0.0095** | **+0.0089** | **+0.0097** | **+0.0038** |

### Evaluation & Discussion
- The derived features produce a **consistent, verified improvement** across every single evaluation metric on the untouched test partition.
- **F1-Score Gain:** F1 rose from **0.6601 to 0.6696** (+0.0095 points / ~1.44% relative gain).
- **Accuracy Gain:** Accuracy rose from **85.40% to 85.78%** (+0.38 percentage points).
- **Mechanism of Improvement:** `capital_net` collapses collinear gain/loss dynamics into a single net asset balance. `hours_per_age` differentiates young individuals working 40 hours early in career from senior individuals nearing retirement.

---

## 6. Label-Quality Audit Findings

### 1. Duplicate Feature Vectors with Conflicting Income Labels
The audit revealed **10 records across 5 distinct feature combinations** where individuals share 100% identical demographic and employment attributes, but have contradictory income classifications (`<=50K` vs `>50K`). The actual records from `results/label_audit_conflicts.csv` are summarized below:

| Conflict Group | Dataset Indices | Age | Workclass | fnlwgt | Education | Marital-Status | Occupation | Relationship | Race | Sex | Hours | Native-Country | Conflicting Incomes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `CONFLICT_GROUP_01` | 1781, 32964 | 46 | Private | 235646 | HS-grad (9) | Married-civ-spouse | Craft-repair | Husband | White | Male | 40 | United-States | `<=50K` (Row 1781) vs `>50K` (Row 32964) |
| `CONFLICT_GROUP_02` | 45787, 2373 | 34 | Private | 169583 | Masters (14) | Never-married | Exec-managerial | Not-in-family | White | Female | 50 | United-States | `<=50K` (Row 45787) vs `>50K` (Row 2373) |
| `CONFLICT_GROUP_03` | 11692, 16161 | 41 | Private | 242804 | HS-grad (9) | Married-civ-spouse | Transport-moving | Husband | White | Male | 40 | United-States | `<=50K` (Row 11692) vs `>50K` (Row 16161) |
| `CONFLICT_GROUP_04` | 27210, 15802 | 46 | Private | 224559 | HS-grad (9) | Married-civ-spouse | Transport-moving | Husband | Black | Male | 40 | United-States | `<=50K` (Row 27210) vs `>50K` (Row 15802) |
| `CONFLICT_GROUP_05` | 17129, 39042 | 39 | Private | 138192 | Bachelors (13) | Married-civ-spouse | Craft-repair | Husband | White | Male | 40 | United-States | `<=50K` (Row 17129) vs `>50K` (Row 39042) |

#### Data Engineering Interpretation & Limitations
For each pair, all 14 input predictor features match exactly, while the income targets disagree. Identical observed features with conflicting labels establish an empirical label conflict, but do not determine which label is correct. These conflicts arise naturally in observational data from unobserved variables—such as regional cost-of-living differences, specific industry niches, commissions, or equity compensation—that influence true earnings. In accordance with strict data integrity standards, neither label was assumed correct, original labels were preserved, and no records were discarded based on this conflict audit.

### 2. 50-Record Manual Review Sampling
- **Sample Generation:** 50 rows were randomly sampled using fixed `random_state=42` and exported to `results/sampled_50_rows.csv`.
- **Review Checklist:** Documented in detail in `results/label_audit_checklist.md`.
- **Key Review Outcomes:**
  - **Range & Type Conformance:** 100% of sampled rows fall strictly within valid ranges (`age` $\in [17, 90]$, `hours-per-week` $\in [1, 99]$, `capital-gain` $\in [0, 99999]$, `capital-loss` $\in [0, 4356]$) and allowed categorical domain vocabularies.
  - **Missing Value Tracking:** 3 records exhibit expected missing data markers: Row `21063` (missing `workclass` and `occupation`), Row `20139` (missing `native-country`), and Row `18957` (missing `workclass` and `occupation`).
  - **Unusual but Permissible Values:** Notable working intensities include part-time roles (4 h/wk in Row `3375`, 15 h/wk in Row `32507`) and high weekly commitments (70 h/wk in Row `1276`, 75 h/wk in Row `43769`, 84 h/wk in Row `3136`). All represent plausible real-world employment patterns rather than data corruption.
  - **Label Policy & Limitation:** True income was not inferred or predicted from demographic proxies (such as education, gender, or race). Ground-truth verification against confidential 1994 Census Bureau interview records remains impossible with publicly available data. All original labels were preserved untouched.

---

## 7. Cryptographic Reproducibility Manifest

All raw and processed artifacts were hashed using SHA256:

- **Raw Parquet SHA256:** `23244698d388a08162887b859c35d3f3f45a712aa5ffe5a2034add94ad77b32c`
- **Training Set (Deduplicated) SHA256:** `c23b14491ccd173539d45cb53bfc953e6657aad07a9d8b6a90c49979f8906d3b`
- **Test Set SHA256:** `4fff27331581e54160031e81bc5c826c7ac9e0c39d44398926d28bab6b659285`
- **Fitted Preprocessor Joblib SHA256:** `6b62edcc72fad876f7731cef938e4e8ebfa4a91c90ba7ac562ad5b185a89721c`
- Complete environment details (Python 3.11.8, scikit-learn 1.5.0, pandas 2.2.2, numpy 1.26.4) are permanently logged in `results/reproducibility_manifest.json`.

The manual review of 50 sampled records found no obvious violations of the specified numeric ranges. Missing values were identified and documented. The duplicate-feature conflict audit detected five feature combinations (10 records) with conflicting income labels. These records were flagged for investigation without changing their original labels. The manual review does not independently establish the correctness of the source income labels.
