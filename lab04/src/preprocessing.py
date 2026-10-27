"""

Leakage-Free Preprocessing Module for UCI Adult Dataset.



Implements:

1. Stratified 70/30 train/test splitting based on the income target.

2. Training-only exact duplicate removal, leaving test set untouched.

3. Distribution shift tracking caused by training-only deduplication.

4. Custom NumericOutlierClipper fitted strictly on training data percentiles (1st and 99th).

5. Median imputation for numeric features and mode (most frequent) imputation for categorical features.

6. Target leakage prevention: target column is completely excluded from feature transformers.

7. Joblib serialization and deserialization of the fitted preprocessor.

"""



import logging

import sys

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple



import joblib

import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, StandardScaler



logger = logging.getLogger(__name__)





class NumericOutlierClipper(BaseEstimator, TransformerMixin):

    """

    Clips numeric values to lower and upper percentiles learned exclusively

    from the training set.

    """



    def __init__(self, lower_percentile: float = 1.0, upper_percentile: float = 99.0):

        self.lower_percentile = lower_percentile

        self.upper_percentile = upper_percentile

        self.bounds_: Dict[int, Tuple[float, float]] = {}



    def fit(self, X, y=None):

        X_mat = np.asarray(X)

        self.bounds_ = {}

        for col_idx in range(X_mat.shape[1]):

            col_vals = X_mat[:, col_idx]

            # Ignore NaNs during quantile computation if not yet imputed

            low = float(np.nanpercentile(col_vals, self.lower_percentile))

            high = float(np.nanpercentile(col_vals, self.upper_percentile))

            self.bounds_[col_idx] = (low, high)

        return self



    def transform(self, X):

        X_mat = np.asarray(X).copy()

        for col_idx, (low, high) in self.bounds_.items():

            if col_idx < X_mat.shape[1]:

                X_mat[:, col_idx] = np.clip(X_mat[:, col_idx], low, high)

        return X_mat





def perform_stratified_split(

    df: pd.DataFrame,

    config: Dict[str, Any]

) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """

    Split the dataset into 70% train and 30% test using stratified sampling

    on the target column.



    Args:

        df: Input DataFrame containing features and target column.

        config: Configuration dictionary.



    Returns:

        Tuple of (train_df, test_df).

    """

    split_cfg = config.get("split", {})

    target_col = config.get("dataset", {}).get("target_col", "income")

    test_size = split_cfg.get("test_size", 0.3)

    random_state = split_cfg.get("random_state", 42)



    logger.info(

        f"Performing stratified {100*(1-test_size):.0f}/{100*test_size:.0f} split "

        f"on '{target_col}' with random_state={random_state}..."

    )



    train_df, test_df = train_test_split(

        df,

        test_size=test_size,

        stratify=df[target_col],

        random_state=random_state

    )



    logger.info(f"Split completed: Train rows={len(train_df):,}, Test rows={len(test_df):,}")

    return train_df.copy(), test_df.copy()





def deduplicate_train_only(

    train_df: pd.DataFrame,

    target_col: str = "income"

) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    """

    Remove exact duplicate records from the training set only.



    The test set remains completely untouched.

    Documents the potential distribution shift in target proportions caused

    by dropping duplicates after stratified splitting.



    Args:

        train_df: Raw training split.

        target_col: Name of the income target column.



    Returns:

        Tuple of (deduplicated_train_df, shift_audit_metadata).

    """

    initial_count = len(train_df)

    initial_proportions = train_df[target_col].value_counts(normalize=True).to_dict()



    duplicates_mask = train_df.duplicated()

    num_duplicates = int(duplicates_mask.sum())



    dedup_train_df = train_df.drop_duplicates().copy()

    final_count = len(dedup_train_df)

    final_proportions = dedup_train_df[target_col].value_counts(normalize=True).to_dict()



    # Calculate shift for each class

    shift_details = {}

    for cls, initial_p in initial_proportions.items():

        final_p = final_proportions.get(cls, 0.0)

        shift_details[str(cls)] = {

            "initial_proportion": round(float(initial_p), 6),

            "final_proportion": round(float(final_p), 6),

            "absolute_shift": round(float(final_p - initial_p), 6)

        }



    audit_meta = {

        "initial_train_rows": initial_count,

        "exact_duplicates_removed": num_duplicates,

        "final_train_rows": final_count,

        "proportions_before": {str(k): round(float(v), 6) for k, v in initial_proportions.items()},

        "proportions_after": {str(k): round(float(v), 6) for k, v in final_proportions.items()},

        "class_distribution_shift": shift_details

    }



    logger.info(

        f"Training deduplication: removed {num_duplicates} exact duplicate rows "

        f"({initial_count} -> {final_count})."

    )

    for cls, info in shift_details.items():

        logger.info(

            f"  Class '{cls}' shift: {info['initial_proportion']*100:.3f}% -> "

            f"{info['final_proportion']*100:.3f}% (delta: {info['absolute_shift']*100:+.4f}%)"

        )



    return dedup_train_df, audit_meta





def build_preprocessor(

    numeric_cols: List[str],

    categorical_cols: List[str],

    config: Dict[str, Any]

) -> ColumnTransformer:

    """

    Build a leakage-free preprocessing ColumnTransformer.



    Numeric pipeline:

      1. SimpleImputer(strategy='median')

      2. NumericOutlierClipper(1.0, 99.0)

      3. StandardScaler()



    Categorical pipeline:

      1. SimpleImputer(strategy='most_frequent')

      2. OneHotEncoder(handle_unknown='ignore', sparse_output=False)



    Args:

        numeric_cols: List of numeric feature names.

        categorical_cols: List of categorical feature names.

        config: Configuration dictionary.



    Returns:

        Unfitted ColumnTransformer instance.

    """

    prep_cfg = config.get("preprocessing", {})

    num_strategy = prep_cfg.get("numeric_impute_strategy", "median")

    cat_strategy = prep_cfg.get("categorical_impute_strategy", "most_frequent")

    outlier_cfg = prep_cfg.get("outlier_percentiles", {})

    lower_pct = outlier_cfg.get("lower", 1.0)

    upper_pct = outlier_cfg.get("upper", 99.0)



    from sklearn.pipeline import Pipeline



    num_pipeline = Pipeline([

        ("imputer", SimpleImputer(strategy=num_strategy)),

        ("clipper", NumericOutlierClipper(lower_percentile=lower_pct, upper_percentile=upper_pct)),

        ("scaler", StandardScaler())

    ])



    cat_pipeline = Pipeline([

        ("imputer", SimpleImputer(strategy=cat_strategy)),

        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))

    ])



    preprocessor = ColumnTransformer(

        transformers=[

            ("num", num_pipeline, numeric_cols),

            ("cat", cat_pipeline, categorical_cols)

        ]

    )



    return preprocessor





def save_preprocessor(preprocessor: ColumnTransformer, output_path: Path) -> None:

    """Save fitted preprocessor using joblib."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, output_path)

    logger.info(f"Saved fitted preprocessor to {output_path}")





def load_preprocessor(input_path: Any) -> ColumnTransformer:

    """

    Load fitted preprocessor using joblib.



    Robust to execution from either the repository root or inside lab04/.



    Args:

        input_path: Path to preprocessor.joblib (string or Path).



    Returns:

        Loaded ColumnTransformer instance.

    """

    input_path = Path(input_path)

    if not input_path.exists():

        raise FileNotFoundError(f"Preprocessor file not found at {input_path}")



    # Ensure lab04 root is on sys.path so 'src' is resolvable

    lab_dir = Path(__file__).resolve().parent.parent

    if str(lab_dir) not in sys.path:

        sys.path.insert(0, str(lab_dir))



    # Ensure module aliases exist in sys.modules

    if "src.preprocessing" not in sys.modules:

        import src.preprocessing

        sys.modules["src.preprocessing"] = src.preprocessing



    preprocessor = joblib.load(input_path)

    logger.info(f"Loaded preprocessor from {input_path}")

    return preprocessor
