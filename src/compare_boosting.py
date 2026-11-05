
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.base import clone

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT / "data" / "campus_recruitment.csv")
df = df.drop_duplicates().copy()

df["PlacementStatus"] = (
    df["PlacementStatus"].astype(str).str.strip().str.lower()
    .map({"placed": 1, "notplaced": 0})
)

if df["PlacementStatus"].isna().any():
    raise ValueError("Unexpected placement labels.")

X = df.drop(columns="PlacementStatus")
y = df["PlacementStatus"].astype(int)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

# HistGradientBoosting requires dense numeric input.
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ]), cat_cols),
])

models = {
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
}

results = []

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", estimator),
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_valid)
    prob = pipeline.predict_proba(X_valid)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_valid, pred),
        "F1": f1_score(y_valid, pred),
        "ROC_AUC": roc_auc_score(y_valid, prob),
    })

print(pd.DataFrame(results).sort_values(
    "ROC_AUC", ascending=False
).to_string(index=False))
