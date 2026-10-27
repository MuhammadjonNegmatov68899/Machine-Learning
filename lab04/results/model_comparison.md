# Model Comparison: Baseline vs. Feature Engineered Logistic Regression

| Model                                              |   Features Count |   Accuracy |   F1-Score |   Precision |   Recall |   ROC-AUC |
|:---------------------------------------------------|-----------------:|-----------:|-----------:|------------:|---------:|----------:|
| Baseline (Without Derived Features)                |               14 |     0.854  |     0.6601 |      0.7448 |   0.5927 |    0.9058 |
| With Derived Features (capital_net, hours_per_age) |               16 |     0.8578 |     0.6696 |      0.7537 |   0.6024 |    0.9096 |
| Difference (Engineered - Baseline)                 |               +2 |     0.0038 |     0.0095 |      0.0089 |   0.0097 |    0.0038 |

## Analysis of Metric Changes
- **F1-Score:** Improved by **+0.0095** (0.6601 -> 0.6696).
- **Accuracy:** Improved by **+0.0038** (0.854 -> 0.8578).
- **Interpretation:** The derived features (`capital_net` and `hours_per_age`) provide additional domain-specific signal that allows the linear decision boundary to separate high-income individuals more effectively, evidenced by concurrent gains in Accuracy, Precision, Recall, and ROC-AUC without any target leakage.
