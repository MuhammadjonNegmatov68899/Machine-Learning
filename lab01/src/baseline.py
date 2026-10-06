import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Split data: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# Train Logistic Regression
logistic = LogisticRegression(
    max_iter=1000,
    random_state=42
)
logistic.fit(X_train, y_train)

# Train Random Forest
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
random_forest.fit(X_train, y_train)

# Calculate test accuracy
logistic_accuracy = logistic.score(X_test, y_test)
random_forest_accuracy = random_forest.score(X_test, y_test)

# Save results
results = pd.DataFrame({
    "model": ["LogisticRegression", "RandomForestClassifier"],
    "test_accuracy": [
        round(logistic_accuracy, 4),
        round(random_forest_accuracy, 4)
    ]
})

results.to_csv(
    "lab01/results/baseline_accuracy.csv",
    index=False
)

print(results)
