import time
import os
import joblib
import pandas as pd
import random
import numpy as np
import torch

from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# Load dataset
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)


def create_logistic():
    return LogisticRegression(max_iter=1000, random_state=42)


def create_random_forest():
    return RandomForestClassifier(n_estimators=100, random_state=42)


def train_model(model):
    model.fit(X_train, y_train)
    return model


def measure_training_time(create_model):
    model = create_model()
    train_model(model)

    times = []

    for _ in range(5):
        model = create_model()

        start = time.perf_counter()
        train_model(model)
        end = time.perf_counter()

        times.append(end - start)

    times.sort()

    return model, times[len(times) // 2]


def measure_inference_time(model):
    sample = X_test[:1]

    times = []

    for _ in range(100):
        start = time.perf_counter()
        model.predict(sample)
        end = time.perf_counter()

        times.append(end - start)

    times.sort()

    return times[len(times) // 2]


def measure_training_memory(create_model):
    def train():
        model = create_model()
        model.fit(X_train, y_train)

    return memory_usage(
        train,
        interval=0.01,
        max_usage=True
    )


def measure_inference_memory(model):
    sample = X_test[:1]

    def infer():
        for _ in range(100):
            model.predict(sample)

    return memory_usage(
        infer,
        interval=0.01,
        max_usage=True
    )


def save_model_and_get_size(model, filename):
    path = f"lab01/results/{filename}"

    joblib.dump(model, path)

    size_bytes = os.path.getsize(path)
    size_kb = size_bytes / 1024

    return size_bytes, size_kb


def main():

    # Logistic Regression
    logistic_model, logistic_train_time = measure_training_time(
        create_logistic
    )

    logistic_inference_time = measure_inference_time(
        logistic_model
    )

    logistic_training_memory = measure_training_memory(
        create_logistic
    )

    logistic_inference_memory = measure_inference_memory(
        logistic_model
    )

    logistic_peak_memory = max(
        logistic_training_memory,
        logistic_inference_memory
    )

    logistic_bytes, logistic_kb = save_model_and_get_size(
        logistic_model,
        "logistic_regression.joblib"
    )


    # Random Forest
    rf_model, rf_train_time = measure_training_time(
        create_random_forest
    )

    rf_inference_time = measure_inference_time(
        rf_model
    )

    rf_training_memory = measure_training_memory(
        create_random_forest
    )

    rf_inference_memory = measure_inference_memory(
        rf_model
    )

    rf_peak_memory = max(
        rf_training_memory,
        rf_inference_memory
    )

    rf_bytes, rf_kb = save_model_and_get_size(
        rf_model,
        "random_forest.joblib"
    )


    # Results
    results = pd.DataFrame({
        "model": [
            "LogisticRegression",
            "RandomForestClassifier"
        ],
        "training_time_median_s": [
            logistic_train_time,
            rf_train_time
        ],
        "inference_time_median_s": [
            logistic_inference_time,
            rf_inference_time
        ],
        "model_size_bytes": [
            logistic_bytes,
            rf_bytes
        ],
        "model_size_kb": [
            logistic_kb,
            rf_kb
        ],
        "peak_memory_training_mb": [
            logistic_training_memory,
            rf_training_memory
        ],
        "peak_memory_inference_mb": [
            logistic_inference_memory,
            rf_inference_memory
        ],
        "peak_memory_mb": [
            logistic_peak_memory,
            rf_peak_memory
        ]
    })

    results.to_csv(
        "lab01/results/measurements.csv",
        index=False
    )

    print(results)


if __name__ == "__main__":
    main()
