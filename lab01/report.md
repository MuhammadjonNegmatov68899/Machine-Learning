# Lab 1 — Baseline Models and Resource Measurement

## Goal

The goal of this lab is to establish baseline machine learning models and measure their accuracy, training time, inference time, model size, and peak memory usage.

## Method

The Breast Cancer Wisconsin Diagnostic dataset was loaded using `sklearn.datasets.load_breast_cancer`.

The dataset was split into 70% training data and 30% test data using stratification and `random_state=42`.

Two baseline models were trained:

- Logistic Regression with `max_iter=1000` and `random_state=42`
- Random Forest Classifier with `n_estimators=100` and `random_state=42`

Training time was measured using a warm-up run followed by five measurements, using the median value.

Single-sample inference time was measured 100 times, using the median value.

Model size was measured from the serialized `.joblib` files. Peak memory usage was measured during model training.

## Results

### Test Accuracy

| Model                    | Test Accuracy |
| ------------------------ | ------------: |
| Logistic Regression      |        0.9415 |
| Random Forest Classifier |        0.9357 |

### Resource Measurements

| Model                    | Training Time (s) | Inference Time (s) | Model Size (KB) | Peak Memory (MB) |
| ------------------------ | ----------------: | -----------------: | --------------: | ---------------: |
| Logistic Regression      |          0.111186 |          0.0000218 |           1.077 |          146.281 |
| Random Forest Classifier |          0.067059 |           0.001002 |         284.087 |          145.953 |

### Deployment Fit

| Model                    | Cloud | Edge | Mobile | TinyML |
| ------------------------ | ----- | ---- | ------ | ------ |
| Logistic Regression      | Yes   | Yes  | Yes    | No     |
| Random Forest Classifier | Yes   | Yes  | Yes    | No     |

Both models fit the Cloud, Edge, and Mobile resource budgets based on the measured memory, latency, and model size. Neither model fits the TinyML budget because the measured peak memory usage is much greater than 256 KB. The Random Forest model also exceeds the TinyML model-size limit of 100 KB.

## Conclusions

1. Logistic Regression achieved slightly higher test accuracy than Random Forest on the Breast Cancer dataset.

2. Random Forest had a shorter median training time, while Logistic Regression had much faster single-sample inference.

3. Both models are small enough for Cloud, Edge, and Mobile deployment based on the measured resource requirements, but neither is suitable for TinyML without further optimization.
