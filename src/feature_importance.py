
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

model_path = ROOT / "models" / "tuned_random_forest.joblib"
model = joblib.load(model_path)

predictions = model.predict(X_valid)
print("Validation accuracy:", round(accuracy_score(y_valid, predictions), 4))

# Measure the effect of shuffling each original feature
importance = permutation_importance(
    model,
    X_valid,
    y_valid,
    scoring="accuracy",
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance.importances_mean,
    "Std": importance.importances_std,
}).sort_values("Importance", ascending=False)

print("\nPermutation feature importance:")
print(importance_df.to_string(index=False))

output_path = ROOT / "outputs" / "feature_importance.csv"
output_path.parent.mkdir(exist_ok=True)
importance_df.to_csv(output_path, index=False)

# Save a feature-importance chart
plot_data = importance_df.sort_values("Importance")

plt.figure(figsize=(9, 6))
plt.barh(plot_data["Feature"], plot_data["Importance"], xerr=plot_data["Std"])
plt.xlabel("Decrease in validation accuracy after shuffling")
plt.title("Student Placement: Permutation Feature Importance")
plt.tight_layout()

chart_path = ROOT / "outputs" / "feature_importance.png"
plt.savefig(chart_path, dpi=150)
plt.close()

print("\nSaved results to:", output_path)
print("Saved chart to:", chart_path)
