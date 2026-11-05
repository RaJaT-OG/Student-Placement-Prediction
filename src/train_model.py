
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

train_path = OUTPUT_DIR / "placement_train_processed.csv"
valid_path = OUTPUT_DIR / "placement_validation_processed.csv"

if not train_path.exists() or not valid_path.exists():
    raise FileNotFoundError(
        "Processed files not found. Run src/preprocessing.py first."
    )

train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)

X_train = train_df.drop(columns=["placed"])
y_train = train_df["placed"]

X_valid = valid_df.drop(columns=["placed"])
y_valid = valid_df["placed"]

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_valid, predictions),
        "Precision": precision_score(
            y_valid, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_valid, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_valid, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_valid, probabilities),
    }

    results.append(metrics)

    print("\nClassification report:")
    print(classification_report(y_valid, predictions, zero_division=0))

    joblib.dump(model, MODEL_DIR / f"{name.lower().replace(' ', '_')}.pkl")

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))
print("\nModels saved in:", MODEL_DIR)
print("Comparison saved to:", OUTPUT_DIR / "model_comparison.csv")

