"""

Reproducibility and Integrity Verification Module.



Computes cryptographic SHA256 hashes of raw and processed artifacts,

records system environments and package versions, and serializes a complete

reproducibility manifest for auditability.

"""



import hashlib

import json

import logging

import platform

import sys

from pathlib import Path

from typing import Any, Dict, Optional



import numpy as np

import pandas as pd



logger = logging.getLogger(__name__)





def compute_file_sha256(file_path: Path) -> str:

    """Compute the cryptographic SHA256 hex digest of a file on disk."""

    if not file_path.exists():

        return "FILE_NOT_FOUND"



    sha = hashlib.sha256()

    with open(file_path, "rb") as f:

        while chunk := f.read(65536):

            sha.update(chunk)

    return sha.hexdigest()





def compute_dataframe_sha256(df: pd.DataFrame) -> str:

    """

    Compute a deterministic SHA256 hash of a pandas DataFrame.



    Converts DataFrame to deterministic CSV bytes with sorted index and columns.

    """

    # Sort columns alphabetically to ensure deterministic serialization

    sorted_df = df.reindex(sorted(df.columns), axis=1)

    csv_bytes = sorted_df.to_csv(index=True).encode("utf-8")

    return hashlib.sha256(csv_bytes).hexdigest()





def compute_array_sha256(arr: np.ndarray) -> str:

    """Compute a deterministic SHA256 hash of a NumPy array."""

    # Ensure C-contiguous array for consistent memory layout

    arr_c = np.ascontiguousarray(arr)

    return hashlib.sha256(arr_c.tobytes()).hexdigest()





def collect_environment_metadata() -> Dict[str, Any]:

    """Capture runtime environment, Python versions, and exact library dependencies."""

    packages = [

        "numpy",

        "pandas",

        "sklearn",

        "scipy",

        "yaml",

        "joblib",

        "pytest",

    ]

    package_versions = {}

    for pkg in packages:

        try:

            mod = __import__(pkg)

            version = getattr(mod, "__version__", "unknown")

            package_versions[pkg] = version

        except ImportError:

            package_versions[pkg] = "not_installed"



    return {

        "python_version": sys.version,

        "platform": platform.platform(),

        "processor": platform.processor(),

        "package_versions": package_versions

    }





def generate_reproducibility_manifest(

    config: Dict[str, Any],

    raw_df: pd.DataFrame,

    train_df: pd.DataFrame,

    test_df: pd.DataFrame,

    base_dir: Optional[Path] = None

) -> Dict[str, Any]:

    """

    Build and save comprehensive reproducibility manifest.



    Args:

        config: Configuration dictionary.

        raw_df: Standardized raw DataFrame.

        train_df: Deduplicated training DataFrame.

        test_df: Untouched test DataFrame.

        base_dir: Base directory for paths.



    Returns:

        Manifest dictionary.

    """

    if base_dir is None:

        base_dir = Path(__file__).resolve().parent.parent

    else:

        base_dir = Path(base_dir)



    results_cfg = config.get("results", {})

    manifest_path = base_dir / results_cfg.get(

        "reproducibility_manifest_json", "results/reproducibility_manifest.json"

    )



    raw_parquet_path = base_dir / config.get("dataset", {}).get(

        "raw_data_path", "data/raw/adult_raw.parquet"

    )

    raw_csv_path = base_dir / config.get("dataset", {}).get(

        "raw_data_csv", "data/raw/adult_raw.csv"

    )

    preprocessor_path = base_dir / config.get("preprocessing", {}).get(

        "preprocessor_save_path", "results/preprocessor.joblib"

    )



    manifest: Dict[str, Any] = {

        "dataset_name": config.get("dataset", {}).get("name", "adult"),

        "random_seeds": {

            "split_random_state": config.get("split", {}).get("random_state", 42),

            "model_random_state": config.get("model", {}).get("random_state", 42),

            "audit_random_state": config.get("audit", {}).get("random_state", 42),

        },

        "artifact_hashes": {

            "raw_parquet_file_sha256": compute_file_sha256(raw_parquet_path),

            "raw_csv_file_sha256": compute_file_sha256(raw_csv_path),

            "raw_dataframe_sha256": compute_dataframe_sha256(raw_df),

            "train_dataframe_dedup_sha256": compute_dataframe_sha256(train_df),

            "test_dataframe_sha256": compute_dataframe_sha256(test_df),

            "preprocessor_joblib_sha256": compute_file_sha256(preprocessor_path),

        },

        "dataset_shapes": {

            "raw_shape": list(raw_df.shape),

            "train_dedup_shape": list(train_df.shape),

            "test_shape": list(test_df.shape),

        },

        "environment": collect_environment_metadata(),

    }



    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as f:

        json.dump(manifest, f, indent=2)



    logger.info(f"Reproducibility manifest successfully written to {manifest_path}")

    return manifest
