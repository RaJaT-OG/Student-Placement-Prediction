
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "student_placement_train.csv")
df = df.dropna(subset=["placed"]).copy()

y = df["placed"].astype(int)
X = df.drop(columns=["placed", "student_id"], errors="ignore")
X = X.dropna(axis=1, how="all")

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

transformers = []

if numeric_cols:
    transformers.append((
        "num",
        SimpleImputer(strategy="median"),
        numeric_cols
    ))

if categorical_cols:
    transformers.append((
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_cols
    ))

preprocessor = ColumnTransformer(transformers=transformers)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_valid)[:, 1]
real_auc = roc_auc_score(y_valid, probabilities)

# Compare predictions with shuffled validation labels.
# This checks whether the observed AUC is meaningfully above chance.
rng = np.random.default_rng(42)
shuffled_aucs = []

for _ in range(100):
    shuffled_y = rng.permutation(y_valid.to_numpy())
    shuffled_aucs.append(
        roc_auc_score(shuffled_y, probabilities)
    )

print("Real validation ROC-AUC:", round(real_auc, 4))
print("Mean shuffled-label ROC-AUC:", round(np.mean(shuffled_aucs), 4))
print("Shuffled-label 95th percentile:",
      round(np.percentile(shuffled_aucs, 95), 4))

plt.figure(figsize=(7, 4))
plt.hist(shuffled_aucs, bins=20, edgecolor="black")
plt.axvline(real_auc, linestyle="--", label=f"Real AUC = {real_auc:.3f}")
plt.xlabel("ROC-AUC with shuffled validation labels")
plt.ylabel("Frequency")
plt.title("Placement Model: Chance-Level Comparison")
plt.legend()
plt.tight_layout()

output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / "target_analysis.png", dpi=150)
plt.show()
