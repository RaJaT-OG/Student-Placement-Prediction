
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "campus_recruitment.csv")
df = df.drop_duplicates().copy()

df["PlacementStatus"] = (
    df["PlacementStatus"].astype(str).str.strip().str.lower()
    .map({"placed": 1, "notplaced": 0})
)

if df["PlacementStatus"].isna().any():
    raise ValueError("Unexpected placement labels.")

X = df.drop(columns=["PlacementStatus"])
y = df["PlacementStatus"].astype(int)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
])

param_distributions = {
    "model__n_estimators": [200, 300, 500, 700],
    "model__max_depth": [None, 8, 12, 16, 20, 25],
    "model__min_samples_split": [2, 5, 10, 15],
    "model__min_samples_leaf": [1, 2, 3, 5, 8],
    "model__max_features": ["sqrt", 0.7, 1.0],
    "model__class_weight": [None, "balanced"],
}

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=30,
    scoring="accuracy",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

search.fit(X_train, y_train)

best_model = search.best_estimator_
predictions = best_model.predict(X_valid)
probabilities = best_model.predict_proba(X_valid)[:, 1]

print("\nBest cross-validation accuracy:", round(search.best_score_, 4))
print("Best parameters:", search.best_params_)
print("Validation accuracy:", round(accuracy_score(y_valid, predictions), 4))
print("Validation F1:", round(f1_score(y_valid, predictions), 4))
print("Validation ROC-AUC:", round(roc_auc_score(y_valid, probabilities), 4))
print("\nClassification report:")
print(classification_report(
    y_valid, predictions,
    target_names=["NotPlaced", "Placed"],
    zero_division=0,
))

model_path = ROOT / "models" / "tuned_random_forest.joblib"
model_path.parent.mkdir(exist_ok=True)
joblib.dump(best_model, model_path)
print("\nSaved tuned model to:", model_path)
