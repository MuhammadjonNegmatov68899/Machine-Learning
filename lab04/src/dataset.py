"""

Dataset Ingestion and Management Module for UCI Adult Dataset.



Handles fetching from OpenML, raw data persistence, dimension validation,

and separation of the 14 input features from the target column.

"""



import logging

import os

from pathlib import Path

from typing import Dict, Optional, Tuple, Any



import pandas as pd

from sklearn.datasets import fetch_openml



logger = logging.getLogger(__name__)





def ingest_raw_data(

    config: Dict[str, Any],

    base_dir: Optional[Path] = None,

    force_download: bool = False

) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    """

    Ingest the raw UCI Adult dataset from OpenML (v2).



    Preserves raw data without modifications, verifies row and column dimensions,

    and caches raw files locally under data/raw/.



    Args:

        config: Configuration dictionary (loaded from config.yaml).

        base_dir: Base directory for relative paths (default: parent of src/).

        force_download: If True, re-fetches from OpenML even if local raw files exist.



    Returns:

        Tuple of (raw_dataframe, ingestion_metadata).



    Raises:

        ValueError: If actual dimensions differ from expected dimensions.

    """

    if base_dir is None:

        base_dir = Path(__file__).resolve().parent.parent

    else:

        base_dir = Path(base_dir)



    dataset_cfg = config.get("dataset", {})

    raw_path_parquet = base_dir / dataset_cfg.get("raw_data_path", "data/raw/adult_raw.parquet")

    raw_path_csv = base_dir / dataset_cfg.get("raw_data_csv", "data/raw/adult_raw.csv")



    raw_path_parquet.parent.mkdir(parents=True, exist_ok=True)



    # 1. Ingestion / Fetching

    if raw_path_parquet.exists() and not force_download:

        try:

            logger.info(f"Loading raw dataset from local parquet cache: {raw_path_parquet}")

            df_raw = pd.read_parquet(raw_path_parquet)

            source = "local_cache_parquet"

        except Exception:

            if raw_path_csv.exists():

                logger.info(f"Loading raw dataset from local csv cache: {raw_path_csv}")

                df_raw = pd.read_csv(raw_path_csv)

                source = "local_cache_csv"

            else:

                raise

    elif raw_path_csv.exists() and not force_download:

        logger.info(f"Loading raw dataset from local csv cache: {raw_path_csv}")

        df_raw = pd.read_csv(raw_path_csv)

        source = "local_cache_csv"

    else:

        dataset_name = dataset_cfg.get("name", "adult")

        dataset_version = dataset_cfg.get("version", 2)

        logger.info(f"Fetching '{dataset_name}' (version {dataset_version}) from OpenML...")



        bunch = fetch_openml(name=dataset_name, version=dataset_version, as_frame=True)

        df_raw = bunch.frame.copy()

        source = f"openml:{dataset_name}:v{dataset_version}"



        # Save exact raw copy to csv (always supported) and parquet (if pyarrow available)

        df_raw.to_csv(raw_path_csv, index=False)

        try:

            df_raw.to_parquet(raw_path_parquet, index=False)

            logger.info(f"Persisted exact raw data to {raw_path_parquet} and {raw_path_csv}")

        except Exception as e:

            logger.warning(f"Parquet engine not available ({e}). Preserved raw data to {raw_path_csv}")



    # 2. Dimension Verification

    actual_rows, actual_cols = df_raw.shape

    expected_rows = dataset_cfg.get("expected_rows", 48842)

    expected_cols = dataset_cfg.get("expected_cols", 15)



    logger.info(f"Raw data dimensions: {actual_rows} rows, {actual_cols} columns")



    dimension_match = (actual_rows == expected_rows) and (actual_cols == expected_cols)

    if not dimension_match:

        discrepancy_msg = (

            f"Dimension discrepancy! Expected ({expected_rows}, {expected_cols}), "

            f"but found ({actual_rows}, {actual_cols})."

        )

        logger.error(discrepancy_msg)

        raise ValueError(discrepancy_msg)



    # 3. Target Standardisation and Metadata

    raw_target_col = dataset_cfg.get("raw_target_col", "class")

    target_col = dataset_cfg.get("target_col", "income")



    # Rename target column for downstream standardization if named 'class'

    df_standardized = df_raw.copy()

    if raw_target_col in df_standardized.columns and raw_target_col != target_col:

        df_standardized = df_standardized.rename(columns={raw_target_col: target_col})

        logger.info(f"Standardized target column '{raw_target_col}' -> '{target_col}'")



    feature_cols = [c for c in df_standardized.columns if c != target_col]



    metadata = {

        "source": source,

        "rows": actual_rows,

        "columns": actual_cols,

        "column_names": list(df_standardized.columns),

        "feature_names": feature_cols,

        "target_name": target_col,

        "raw_target_name": raw_target_col,

        "raw_parquet_path": str(raw_path_parquet),

        "raw_csv_path": str(raw_path_csv),

        "dimension_verified": dimension_match

    }



    return df_standardized, metadata
