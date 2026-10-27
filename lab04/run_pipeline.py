#!/usr/bin/env python3

"""

Machine Learning Lab 04: Main Pipeline Entry Point.



Executes all tasks end-to-end:

1. Ingestion of raw UCI Adult dataset (OpenML v2) with dimension verification.

2. Comprehensive data validation (schema, ranges, missing markers, duplicates).

3. Leakage-free 70/30 stratified splitting and training-only deduplication.

4. Independent preprocessing pipeline fitting and feature engineering.

5. Baseline vs. Feature Engineered Logistic Regression training and evaluation.

6. Label-quality audit (50-record random sample and conflicting label analysis).

7. Cryptographic SHA256 hashing and reproducibility manifest creation.

"""



import argparse

import logging

import sys

from pathlib import Path

from typing import Any, Dict



import yaml



# Ensure lab04 root is on sys.path

LAB_DIR = Path(__file__).resolve().parent

if str(LAB_DIR) not in sys.path:

    sys.path.insert(0, str(LAB_DIR))



from src.dataset import ingest_raw_data

from src.label_audit import run_label_quality_audit

from src.modeling import train_and_compare_models

from src.preprocessing import deduplicate_train_only, perform_stratified_split

from src.reproducibility import generate_reproducibility_manifest

from src.validation import validate_raw_data





def setup_logger(log_level: int = logging.INFO) -> logging.Logger:

    """Configure structured logging to console."""

    logging.basicConfig(

        level=log_level,

        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",

        datefmt="%Y-%m-%d %H:%M:%S",

    )

    return logging.getLogger("lab04_pipeline")





def load_config(config_path: Path) -> Dict[str, Any]:

    """Load configuration YAML file."""

    if not config_path.exists():

        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:

        config = yaml.safe_load(f)

    return config





def run_pipeline(config_path: Path, force_download: bool = False) -> None:

    """Execute the complete data engineering and ML pipeline."""

    logger = setup_logger()

    logger.info("=" * 70)

    logger.info("STARTING LAB 04: UCI ADULT DATA ENGINEERING & ML PIPELINE")

    logger.info("=" * 70)



    # 0. Load Configuration

    config = load_config(config_path)

    logger.info(f"Loaded configuration from: {config_path}")



    # Step 1: Raw Data Ingestion & Dimension Verification

    logger.info("\n>>> STEP 1: RAW DATA INGESTION (Task 1 & Task 2)")

    raw_df, ingestion_meta = ingest_raw_data(

        config=config, base_dir=LAB_DIR, force_download=force_download

    )

    logger.info(

        f"Ingestion successful! Source: {ingestion_meta['source']} | "

        f"Shape: {raw_df.shape[0]:,} rows x {raw_df.shape[1]} columns"

    )



    # Step 2: Data Validation

    logger.info("\n>>> STEP 2: RAW DATA VALIDATION (Task 3)")

    val_report, is_valid = validate_raw_data(

        df=raw_df, config=config, base_dir=LAB_DIR

    )

    logger.info(

        f"Validation completed. All checks passed: {is_valid} | "

        f"Exact Duplicates: {val_report['duplicates']['exact_duplicate_rows']:,}"

    )



    # Step 3: Stratified Splitting & Train Deduplication

    logger.info("\n>>> STEP 3: LEAKAGE-FREE SPLITTING & DEDUPLICATION (Task 4)")

    train_df_raw, test_df = perform_stratified_split(df=raw_df, config=config)



    target_col = config.get("dataset", {}).get("target_col", "income")

    train_df_dedup, dedup_meta = deduplicate_train_only(

        train_df=train_df_raw, target_col=target_col

    )

    logger.info(

        f"Deduplication complete: {dedup_meta['initial_train_rows']:,} -> "

        f"{dedup_meta['final_train_rows']:,} training rows. "

        f"Test set untouched ({len(test_df):,} rows)."

    )



    # Step 4: Modeling & Feature Engineering Baseline Comparison

    logger.info("\n>>> STEP 4: FEATURE ENGINEERING & MODELING COMPARISON (Task 4 & Task 5)")

    comparison_df, full_model_results = train_and_compare_models(

        train_df=train_df_dedup,

        test_df=test_df,

        config=config,

        base_dir=LAB_DIR,

    )

    logger.info("Model Training & Evaluation Summary:")

    print("\n" + comparison_df.to_string(index=False) + "\n")



    # Step 5: Label-Quality Audit

    logger.info("\n>>> STEP 5: LABEL-QUALITY AUDIT (Task 6)")

    sampled_50, conflicting_rows, audit_summary = run_label_quality_audit(

        df=raw_df, config=config, base_dir=LAB_DIR

    )

    logger.info(

        f"Audit complete: Sampled 50 rows saved to {audit_summary['sampled_records_path']}. "

        f"Found {audit_summary['total_conflicting_rows']} rows with conflicting labels across "

        f"{audit_summary['unique_conflicting_feature_vectors']} identical feature combinations."

    )



    # Step 6: Reproducibility & SHA256 Hashes

    logger.info("\n>>> STEP 6: REPRODUCIBILITY & SHA256 HASHES (Task 7)")

    manifest = generate_reproducibility_manifest(

        config=config,

        raw_df=raw_df,

        train_df=train_df_dedup,

        test_df=test_df,

        base_dir=LAB_DIR,

    )

    logger.info(f"Manifest written. Raw Parquet SHA256: {manifest['artifact_hashes']['raw_parquet_file_sha256']}")

    logger.info(f"Train DF SHA256: {manifest['artifact_hashes']['train_dataframe_dedup_sha256']}")

    logger.info(f"Test DF SHA256: {manifest['artifact_hashes']['test_dataframe_sha256']}")



    logger.info("=" * 70)

    logger.info("LAB 04 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")

    logger.info("=" * 70)





def main():

    parser = argparse.ArgumentParser(description="Run Lab 04 Data Engineering & ML Pipeline")

    parser.add_argument(

        "--config",

        type=Path,

        default=LAB_DIR / "config.yaml",

        help="Path to YAML configuration file",

    )

    parser.add_argument(

        "--force-download",

        action="store_true",

        help="Force re-download of dataset from OpenML even if cached",

    )

    args = parser.parse_args()



    try:

        run_pipeline(config_path=args.config, force_download=args.force_download)

    except Exception as e:

        logging.error(f"Pipeline failed with error: {e}", exc_info=True)

        sys.exit(1)





if __name__ == "__main__":

    main()
