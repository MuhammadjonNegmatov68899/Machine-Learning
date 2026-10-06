# Lab 1 — Baseline Models and Resource Measurement

## Goal

The goal of this lab is to establish baseline machine learning models and measure their accuracy, training time, inference time, model size, and peak memory usage.

## Method

The Breast Cancer Wisconsin Diagnostic dataset was loaded using `sklearn.datasets.load_breast_cancer`.

The dataset was split into 70% training data and 30% test data using stratification and `random_state=42`.

Two baseline models were trained:

- Logistic Regression with `max_iter=1000` and `random_state=42`
- Random Forest Classifier with `n_estimators=100` and `random_state=42`

Training time was measured using one warm-up run followed by five measured runs. The median training time was reported.

Single-sample inference time was measured 100 times, and the median was reported.

Model size was measured from the serialized `.joblib` files.

Peak memory usage was measured separately during model training and single-sample inference using `memory_profiler`. The maximum of the two measurements was used as the overall peak memory.

## Results

### Test Accuracy

| Model                    | Test Accuracy |
| ------------------------ | ------------: |
| Logistic Regression      |        0.9415 |
| Random Forest Classifier |        0.9357 |

### Resource Measurements

| Model                    | Training Time (s) | Inference Time (s) | Model Size (KB) | Training Memory (MB) | Inference Memory (MB) | Peak Memory (MB) |
| ------------------------ | ----------------: | -----------------: | --------------: | -------------------: | --------------------: | ---------------: |
| Logistic Regression      |          0.154253 |          0.0000247 |           1.077 |              163.641 |               125.703 |          163.641 |
| Random Forest Classifier |          0.068864 |           0.001149 |         284.087 |              130.734 |                84.516 |          130.734 |

### Deployment Fit

| Model                    | Cloud | Edge | Mobile | TinyML |
| ------------------------ | ----- | ---- | ------ | ------ |
| Logistic Regression      | Yes   | Yes  | Yes    | No     |
| Random Forest Classifier | Yes   | Yes  | Yes    | No     |

Both models fit the Cloud, Edge, and Mobile resource budgets based on the measured memory, latency, and model size. Peak memory for Logistic Regression was measured at 163.641 MB and for Random Forest at 130.734 MB, both fitting within the 256 MB Mobile budget. Neither model fits the TinyML budget because their peak memory usage exceeds 256 KB. Random Forest also exceeds the TinyML model-size limit of 100 KB.

## Conclusions

1. Logistic Regression achieved slightly higher test accuracy than Random Forest on the Breast Cancer Wisconsin Diagnostic dataset.

2. Random Forest had a shorter median training time, while Logistic Regression had much faster single-sample inference.

3. Both models fit the Cloud, Edge, and Mobile deployment budgets, but neither model is suitable for TinyML without further optimization.
