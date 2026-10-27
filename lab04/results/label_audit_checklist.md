# UCI Adult Dataset: 50-Record Manual Review Checklist

> **Audit Methodology & Verification Standards:**
> This review covers a reproducible random sample of 50 records drawn from the raw UCI Adult dataset (seed 42).
> All records were inspected across automated bounds, schema types, missing value markers, and domain plausibility.
> Per data engineering principles, demographic features are never used to subjectively infer or alter true income labels.

### Review Criteria Applied:
1. **Numeric Range Checks:** Validated age in [17, 90], hours-per-week in [1, 99], capital-gain in [0, 99999], capital-loss in [0, 4356].
2. **Category Validity:** Verified non-missing categorical values against UCI Adult domain standards.
3. **Missing Value Tracking:** Explicitly noted standard NaNs / '?' markers in workclass, occupation, and native-country.
4. **Permissible Extremes:** Distinguished unusual but valid scenarios (e.g. 4 h/wk part-time, 84 h/wk farming/sales) from data corruptions.
5. **Ground-Truth Boundary:** Explicitly acknowledged that original 1994 census interview forms are confidential and cannot be re-polled.

| Sample # | Row Index | Age | Workclass | Education | Occupation | Hours | Cap Gain | Cap Loss | Income | Review Findings | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `7762` | 56 | Private | HS-grad | Other-service | 40 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 2 | `23881` | 25 | Private | HS-grad | Transport-moving | 40 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 3 | `30507` | 43 | Private | Bachelors | Prof-specialty | 40 | 14344 | 0 | **>50K** | Cap-gain \$14344 | [x] Completed (Unusual Valid Value) |
| 4 | `28911` | 32 | Private | HS-grad | Transport-moving | 40 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 5 | `19484` | 39 | Private | Bachelors | Sales | 30 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 6 | `43031` | 20 | Private | HS-grad | Adm-clerical | 40 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 7 | `28188` | 54 | Private | HS-grad | Transport-moving | 45 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 8 | `12761` | 25 | Private | Bachelors | Prof-specialty | 40 | 0 | 1887 | **>50K** | Cap-loss \$1887 | [x] Completed (Unusual Valid Value) |
| 9 | `40834` | 30 | Private | HS-grad | Machine-op-inspct | 50 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 10 | `27875` | 33 | Private | Some-college | Adm-clerical | 40 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 11 | `1276` | 42 | Self-emp-not-inc | Some-college | Farming-fishing | 70 | 0 | 0 | **<=50K** | High intensity (70h/wk) | [x] Completed (Unusual Valid Value) |
| 12 | `22608` | 24 | Private | Bachelors | Adm-clerical | 30 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 13 | `36230` | 48 | Private | Some-college | Craft-repair | 40 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 14 | `13398` | 21 | Private | 11th | Handlers-cleaners | 45 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 15 | `43536` | 37 | State-gov | Bachelors | Prof-specialty | 40 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 16 | `18627` | 36 | Self-emp-not-inc | Assoc-voc | Tech-support | 60 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 17 | `38424` | 26 | Private | Bachelors | Exec-managerial | 45 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 18 | `35505` | 25 | Private | 5th-6th | Machine-op-inspct | 40 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 19 | `2372` | 66 | Self-emp-not-inc | HS-grad | Adm-clerical | 42 | 0 | 0 | **>50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 20 | `3375` | 37 | Private | Assoc-voc | Other-service | 4 | 0 | 0 | **<=50K** | Part-time (4h/wk) | [x] Completed (Unusual Valid Value) |
| 21 | `28331` | 54 | Private | Bachelors | Sales | 38 | 0 | 1408 | **<=50K** | Cap-loss \$1408 | [x] Completed (Unusual Valid Value) |
| 22 | `46820` | 45 | Private | Doctorate | Prof-specialty | 35 | 0 | 3004 | **>50K** | Cap-loss \$3004 | [x] Completed (Unusual Valid Value) |
| 23 | `147` | 33 | Private | HS-grad | Sales | 40 | 0 | 0 | **>50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 24 | `21063` | 67 | [Missing] | Bachelors | [Missing] | 60 | 9386 | 0 | **>50K** | Missing: workclass/occupation; Cap-gain \$9386 | [x] Completed (Missing Documented) |
| 25 | `20139` | 31 | Private | Bachelors | Prof-specialty | 40 | 0 | 0 | **<=50K** | Missing: native-country | [x] Completed (Missing Documented) |
| 26 | `23288` | 22 | Private | Some-college | Adm-clerical | 39 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 27 | `18957` | 62 | [Missing] | Some-college | [Missing] | 24 | 0 | 0 | **<=50K** | Missing: workclass/occupation | [x] Completed (Missing Documented) |
| 28 | `31256` | 22 | State-gov | Some-college | Other-service | 39 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 29 | `29642` | 24 | Local-gov | Bachelors | Prof-specialty | 40 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 30 | `23638` | 32 | Private | 10th | Craft-repair | 40 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 31 | `18067` | 35 | Local-gov | Masters | Prof-specialty | 35 | 0 | 0 | **>50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 32 | `879` | 45 | State-gov | Bachelors | Prof-specialty | 40 | 0 | 0 | **>50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 33 | `7155` | 40 | Private | HS-grad | Machine-op-inspct | 40 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 34 | `43769` | 27 | Private | 5th-6th | Other-service | 75 | 0 | 0 | **<=50K** | High intensity (75h/wk) | [x] Completed (Unusual Valid Value) |
| 35 | `17697` | 32 | Private | HS-grad | Other-service | 52 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 36 | `21167` | 34 | Private | HS-grad | Exec-managerial | 30 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 37 | `3136` | 58 | Self-emp-not-inc | HS-grad | Sales | 84 | 0 | 0 | **<=50K** | High intensity (84h/wk) | [x] Completed (Unusual Valid Value) |
| 38 | `34254` | 25 | Private | HS-grad | Tech-support | 40 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 39 | `17657` | 44 | Self-emp-not-inc | HS-grad | Transport-moving | 55 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 40 | `37254` | 21 | Private | HS-grad | Handlers-cleaners | 36 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 41 | `1422` | 61 | Private | HS-grad | Other-service | 40 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 42 | `32507` | 18 | Private | 11th | Sales | 15 | 0 | 0 | **<=50K** | Part-time (15h/wk) | [x] Completed (Unusual Valid Value) |
| 43 | `30150` | 40 | Private | HS-grad | Craft-repair | 50 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 44 | `14172` | 31 | Private | HS-grad | Machine-op-inspct | 40 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 45 | `32273` | 28 | Private | HS-grad | Handlers-cleaners | 30 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 46 | `48262` | 26 | Private | HS-grad | Other-service | 40 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 47 | `10007` | 46 | Private | 10th | Craft-repair | 40 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 48 | `47649` | 27 | Self-emp-not-inc | Some-college | Transport-moving | 50 | 0 | 0 | **>50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 49 | `43058` | 51 | Private | Assoc-voc | Prof-specialty | 40 | 0 | 0 | **<=50K** | Standard valid values; compliant schema | [x] Completed (Valid) |
| 50 | `708` | 50 | State-gov | Bachelors | Craft-repair | 40 | 0 | 0 | **>50K** | Standard valid values; compliant schema | [x] Completed (Valid) |

## Manual Review Summary & Findings

- **Verification Status:** [x] Completed and Documented for all 50 sampled records.
- **Automated Range & Type Conformance:** 100% of the 50 sampled rows satisfy the specified numeric bounds and categorical types.
- **Missing Value Occurrences:** Exactly 3 records in the sample exhibit expected missingness:
  - Row `21063` (Sample #24): Missing `workclass` and `occupation` (native-country: United-States).
  - Row `20139` (Sample #25): Missing `native-country` (workclass: Private, occupation: Prof-specialty).
  - Row `18957` (Sample #27): Missing `workclass` and `occupation` (native-country: United-States).
- **Unusual Working Intensities Noted:**
  - Part-time ($\le 15$ hrs/wk): Row `3375` (4 hrs/wk, other-service), Row `32507` (15 hrs/wk, age 18 student).
  - High intensity ($\ge 70$ hrs/wk): Row `1276` (70 hrs/wk, farming-fishing), Row `43769` (75 hrs/wk, other-service), Row `3136` (84 hrs/wk, sales).
- **Capital Market Transactions:** 5 records report capital gains or losses (Rows `30507`, `12761`, `28331`, `46820`, `21063`).
- **Label Plausibility & Non-Inference Policy:** Income labels were not altered or inferred based on demographic features. Demographic proxies must never be used to override reported income labels.
- **Verification Limitations:** Independent re-verification against original confidential 1994 U.S. Census Bureau survey returns is impossible with publicly distributed benchmark data.
- **Duplicate-Feature Conflict Findings:** Cross-verified with `results/label_audit_conflicts.csv`. The sample contains no internal conflicts, while the full dataset contains 5 confirmed conflict pairs (10 rows).
