
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "student_placement_train.csv"

df = pd.read_csv(DATA)

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df["placed"].value_counts(dropna=False))

print("\nTarget percentages:")
print((df["placed"].value_counts(normalize=True) * 100).round(2))

print("\nMissing target values:", df["placed"].isna().sum())

print("\nNumeric feature correlations with placement:")
numeric = df.select_dtypes(include="number")
if "placed" in numeric.columns:
    print(
        numeric.corr(numeric_only=True)["placed"]
        .drop("placed")
        .sort_values(key=abs, ascending=False)
        .head(15)
    )
