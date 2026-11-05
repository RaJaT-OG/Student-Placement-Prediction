
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "student_placement_train.csv")

df = df.dropna(subset=["placed"]).copy()

y = df["placed"].astype(int)
X = df.drop(columns=["placed", "student_id"], errors="ignore")

# Remove entirely empty columns
X = X.dropna(axis=1, how="all")

# Detect feature types
numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Preprocessing
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

transformers = []
if numeric_cols:
    transformers.append(("num", numeric_pipe, numeric_cols))
if categorical_cols:
    transformers.append(("cat", categorical_pipe, categorical_cols))

preprocessor = ColumnTransformer(transformers=transformers)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Majority baseline": DummyClassifier(strategy="most_frequent"),
    "Random Forest": Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ]),
}

for name, model in models.items():
    print(f"\n{'=' * 45}\n{name}")
    model.fit(X_train, y_train)

    pred = model.predict(X_valid)
    print("Accuracy:", round(accuracy_score(y_valid, pred), 4))
    print("Precision:", round(
        precision_score(y_valid, pred, zero_division=0), 4
    ))
    print("Recall:", round(
        recall_score(y_valid, pred, zero_division=0), 4
    ))
    print("F1:", round(f1_score(y_valid, pred, zero_division=0), 4))

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_valid)[:, 1]
        print("ROC-AUC:", round(roc_auc_score(y_valid, prob), 4))

    print(classification_report(y_valid, pred, zero_division=0))
