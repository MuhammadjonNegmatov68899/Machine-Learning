# Data Card: UCI Adult Dataset (Census Income)

## 1. Dataset Overview
- **Dataset Name:** Adult (also known as "Census Income" dataset)
- **OpenML Identifier:** `adult` (Version 2)
- **Primary Citation:** Ronny Kohavi and Barry Becker, *"Data Mining and Visualization"*, Silicon Graphics. Available on UCI Machine Learning Repository (1996).
- **Extraction Source:** 1994 United States Census Bureau database (`Current Population Survey`).
- **Target Task:** Binary classification to predict whether an individual's annual income exceeds \$50,000 USD (`<=50K` vs `>50K`).
- **License:** CC BY 4.0 / Public Domain (OpenML & UCI Repository).

---

## 2. Dataset Dimensions and Structure
- **Total Records (Instances):** 48,842 (32,561 historically in `adult.data` + 16,281 in `adult.test`, unified in OpenML v2).
- **Total Attributes (Columns):** 15 (14 input features + 1 target variable).
- **Target Variable:** `income` (originally designated as `class` in OpenML).
  - `<=50K`: 37,155 records (76.07%) — Negative Class (0)
  - `>50K`: 11,687 records (23.93%) — Positive Class (1)
  - **Class Ratio:** ~3.18 to 1 (moderately imbalanced).

---

## 3. Feature Descriptions and Schema

| # | Feature Name | Representation | Type | Valid Range / Categories | Description |
|---|---|---|---|---|---|
| 1 | `age` | Input Feature | Numeric (int) | [17, 90] | Age of the individual in years. |
| 2 | `workclass` | Input Feature | Categorical (8) | Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked | Broad sector of employment. |
| 3 | `fnlwgt` | Input Feature | Numeric (int) | [12,285, 1,490,400] | Final sampling weight calculated by Census Bureau (number of people the record represents). |
| 4 | `education` | Input Feature | Categorical (16) | Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool | Highest educational degree attained. |
| 5 | `education-num` | Input Feature | Numeric (int) | [1, 16] | Number of educational years corresponding to degree. |
| 6 | `marital-status` | Input Feature | Categorical (7) | Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse | Marital status. |
| 7 | `occupation` | Input Feature | Categorical (14) | Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces | Job category or occupation code. |
| 8 | `relationship` | Input Feature | Categorical (6) | Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried | Relationship role within household. |
| 9 | `race` | Input Feature | Categorical (5) | White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black | Racial category. |
| 10 | `sex` | Input Feature | Categorical (2) | Female, Male | Biological sex. |
| 11 | `capital-gain` | Input Feature | Numeric (int) | [0, 99,999] | Capital gains recorded from investments/assets ($). |
| 12 | `capital-loss` | Input Feature | Numeric (int) | [0, 4,356] | Capital losses recorded from investments/assets ($). |
| 13 | `hours-per-week` | Input Feature | Numeric (int) | [1, 99] | Reported average working hours per week. |
| 14 | `native-country` | Input Feature | Categorical (41) | United-States, Mexico, Philippines, Germany, Canada, Puerto-Rico, El-Salvador, India, Cuba, England, Jamaica, South, China, Italy, Dominican-Republic, Vietnam, Guatemala, Japan, Poland, Columbia, Taiwan, Haiti, Iran, Portugal, Nicaragua, Peru, Greece, Ecuador, France, Ireland, Hong, Cambodia, Trinadad&Tobago, Laos, Thailand, Yugoslavia, Outlying-US(Guam-USVI-etc), Hungary, Honduras, Scotland, Holand-Netherlands | Country of birth. |
| 15 | `income` (Target) | Target Label | Categorical (2) | `<=50K`, `>50K` | Binary annual income threshold. |

---

## 4. Missing Value Characteristics
- Missing values in the raw source were encoded as `'?'`.
- OpenML's ARFF parser translates `'?'` into standard null values (`NaN`).
- Three features contain missing values:
  1. `workclass`: 2,799 missing values (5.73%)
  2. `occupation`: 2,809 missing values (5.75%)
  3. `native-country`: 857 missing values (1.75%)
- All other 11 input features and the target contain zero missing entries.

---

## 5. Intended Use and Benchmarking Scope
- **Academic and Educational Use:** Benchmark dataset for tabular binary classification, demonstrating preprocessing, data leakage prevention, handling class imbalance, and feature engineering.
- **Out-of-Scope Use:** Real-world credit underwriting, employment hiring, or algorithmic welfare distribution.

---

## 6. Known Biases and Ethical Considerations
1. **Historical Demographic Bias (1994 United States):**
   - The dataset reflects 1994 US labor and economic dynamics.
   - Significant systemic pay disparities exist along gender and racial lines in the recorded period.
   - Males constitute 66.8% of records and have a substantially higher historical rate of earning >$50K (30.4%) compared to females (10.9%).
2. **Capital Gain/Loss Distortion:**
   - `capital-gain` is right-censored with an artificial cap at $99,999.
   - Over 91% of respondents report 0 capital gain and over 95% report 0 capital loss, creating extreme zero-inflation and severe skewness.
3. **Sampling Weight (`fnlwgt`):**
   - Sampling weights were computed across demographic blocks by the Census Bureau.
   - They represent population expansion factors rather than individual personal attributes and generally add minimal or misleading predictive signal to non-survey-weighted classifiers.

---

## 7. Privacy and Anonymity
- The dataset was created by Barry Becker from public Census bureau extracts.
- Direct identifiers (Name, Social Security Number, Street Address) were suppressed during creation.
- However, combinations of rare demographics (e.g., unique combinations of age, country, occupation, and weights) carry re-identification risks under modern linkage attacks.
