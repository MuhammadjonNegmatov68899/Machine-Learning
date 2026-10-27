"""

Feature Engineering Module for UCI Adult Dataset.



Creates derived domain features:

- capital_net: capital-gain - capital-loss

- hours_per_age: hours-per-week / (age - 16)



Implements robust handling of missing values and safe division with an explicit policy

for non-positive, zero, or missing denominators.

"""



import logging

from typing import Optional

import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin



logger = logging.getLogger(__name__)





def compute_derived_features(

    df: pd.DataFrame,

    age_offset: float = 16.0,

    min_denominator: float = 1.0

) -> pd.DataFrame:

    """

    Compute derived domain features from raw input features.



    Explicit Division Policy:

    -------------------------

    The denominator for hours_per_age is `age - age_offset`.

    In the valid Adult dataset, age >= 17, so `age - 16 >= 1`.

    However, to guard against invalid or corrupted input where `age <= 16`

    or `age is NaN`, non-positive denominators and NaNs are safely replaced

    with `min_denominator` (default: 1.0).

    A warning is logged if any non-positive or NaN denominator is encountered.



    Args:

        df: Input DataFrame containing 'capital-gain', 'capital-loss', 'hours-per-week', 'age'.

        age_offset: Age subtracted to represent working adult years (default: 16.0).

        min_denominator: Minimum safe denominator to prevent zero/negative division (default: 1.0).



    Returns:

        DataFrame with new columns 'capital_net' and 'hours_per_age'.

    """

    df_out = df.copy()



    # Derived Feature 1: capital_net = capital-gain - capital-loss

    gain = pd.to_numeric(df_out["capital-gain"], errors="coerce").fillna(0.0)

    loss = pd.to_numeric(df_out["capital-loss"], errors="coerce").fillna(0.0)

    df_out["capital_net"] = gain - loss



    # Derived Feature 2: hours_per_age = hours-per-week / (age - 16)

    hours = pd.to_numeric(df_out["hours-per-week"], errors="coerce").fillna(0.0)

    age = pd.to_numeric(df_out["age"], errors="coerce")



    raw_denom = age - age_offset



    # Check for invalid denominators

    invalid_mask = (raw_denom <= 0) | raw_denom.isna()

    invalid_count = int(invalid_mask.sum())

    if invalid_count > 0:

        logger.warning(

            f"Found {invalid_count} records with non-positive or NaN denominator (age - {age_offset}). "

            f"Applying safe floor of {min_denominator}."

        )



    safe_denom = np.where(invalid_mask, min_denominator, raw_denom)

    df_out["hours_per_age"] = hours / safe_denom



    return df_out





class AdultFeatureEngineer(BaseEstimator, TransformerMixin):

    """

    Scikit-learn compatible transformer for creating derived features.

    Can be seamlessly integrated into scikit-learn Pipelines.

    """



    def __init__(

        self,

        include_derived: bool = True,

        age_offset: float = 16.0,

        min_denominator: float = 1.0

    ):

        self.include_derived = include_derived

        self.age_offset = age_offset

        self.min_denominator = min_denominator



    def fit(self, X: pd.DataFrame, y=None):

        return self



    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        if not self.include_derived:

            return X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        return compute_derived_features(

            X,

            age_offset=self.age_offset,

            min_denominator=self.min_denominator

        )
