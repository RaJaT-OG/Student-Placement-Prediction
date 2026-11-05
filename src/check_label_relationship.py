
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "student_placement_train.csv")

target = "placed"

print("Rows:", len(df))
print("Unique target values:", sorted(df[target].dropna().unique()))

# Compare feature distributions by placement status
for col in df.select_dtypes(include="number").columns:
    if col in [target, "student_id"]:
        continue

    summary = df.groupby(target)[col].agg(["mean", "std"])
    print(f"\nFeature: {col}")
    print(summary.round(3))

# Check whether rows with identical features have different labels
features = df.drop(
    columns=[target, "student_id"],
    errors="ignore"
)

duplicate_features = features.duplicated(keep=False)

print("\nRows sharing identical feature values:",
      int(duplicate_features.sum()))

if duplicate_features.any():
    conflicting = df.loc[duplicate_features].groupby(
        features.columns.tolist(),
        dropna=False
    )[target].nunique()

    print("Feature combinations with conflicting labels:",
          int((conflicting > 1).sum()))
