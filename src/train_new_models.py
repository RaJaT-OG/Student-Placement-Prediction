
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "campus_recruitment.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load and clean data
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().copy()

TARGET = "PlacementStatus"

df[TARGET] = (
    df[TARGET].astype(str).str.strip().str.lower()
    .map({"placed": 1, "notplaced": 0})
)

if df[TARGET].isna().any():
    raise ValueError("Unexpected values found in PlacementStatus.")

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

# 2. Split before fitting preprocessing
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# 3. Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
}

results = []
best_auc = -1
best_model_name = None
best_pipeline = None

# 4. Train and evaluate
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_valid)
    probabilities = pipeline.predict_proba(X_valid)[:, 1]

    accuracy = accuracy_score(y_valid, predictions)
    precision = precision_score(
        y_valid, predictions, zero_division=0
    )
    recall = recall_score(
        y_valid, predictions, zero_division=0
    )
    f1 = f1_score(y_valid, predictions, zero_division=0)
    auc = roc_auc_score(y_valid, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc,
    })

    print(f"\n{'=' * 55}")
    print(name)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_valid, predictions))
    print("\nClassification report:")
    print(classification_report(
        y_valid,
        predictions,
        target_names=["NotPlaced", "Placed"],
        zero_division=0,
    ))

    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_pipeline = pipeline

# 5. Save results and complete pipeline
results_df = pd.DataFrame(results).sort_values(
    "ROC_AUC", ascending=False
)

results_path = OUTPUT_DIR / "new_model_comparison.csv"
results_df.to_csv(results_path, index=False)

model_path = MODEL_DIR / "best_placement_pipeline.joblib"
joblib.dump(best_pipeline, model_path)

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print(results_df.to_string(index=False))
print(f"\nBest model by validation ROC-AUC: {best_model_name}")
print(f"Saved model: {model_path}")
print(f"Saved metrics: {results_path}")
