
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT / "data" / "campus_recruitment.csv")
df = df.drop_duplicates().copy()

df["PlacementStatus"] = (
    df["PlacementStatus"].astype(str).str.strip().str.lower()
    .map({"placed": 1, "notplaced": 0})
)

if df["PlacementStatus"].isna().any():
    raise ValueError("Unexpected placement labels.")

y = df["PlacementStatus"].astype(int)

X_aptitude = df[["AptitudeTestScore"]]
X_all = df.drop(columns=["PlacementStatus"])

X_train_a, X_valid_a, y_train, y_valid = train_test_split(
    X_aptitude, y, test_size=0.20, random_state=42, stratify=y
)

X_train_all, X_valid_all, _, _ = train_test_split(
    X_all, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Majority baseline": DummyClassifier(strategy="most_frequent"),
    "Aptitude only": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    ),
}

# Full-feature model uses the preprocessing pipeline from your saved file
import joblib
full_model = joblib.load(
    ROOT / "models" / "tuned_random_forest.joblib"
)
models["Full feature Random Forest"] = full_model

for name, model in models.items():
    if name == "Full feature Random Forest":
        model_X_train, model_X_valid = X_train_all, X_valid_all
    else:
        model_X_train, model_X_valid = X_train_a, X_valid_a

    model.fit(model_X_train, y_train)
    pred = model.predict(model_X_valid)
    prob = model.predict_proba(model_X_valid)[:, 1]

    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_valid, pred), 4))
    print("F1-score:", round(f1_score(y_valid, pred), 4))
    print("ROC-AUC:", round(roc_auc_score(y_valid, prob), 4))
