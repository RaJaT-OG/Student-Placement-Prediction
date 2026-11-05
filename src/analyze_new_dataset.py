
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "campus_recruitment.csv"

df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)

# Remove exact duplicate rows
df = df.drop_duplicates().copy()

print("Shape after removing duplicates:", df.shape)
print("Duplicates remaining:", df.duplicated().sum())

# Convert target into binary values for analysis
df["placed_binary"] = (
    df["PlacementStatus"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"placed": 1, "notplaced": 0})
)

if df["placed_binary"].isna().any():
    print("\nUnexpected target labels:")
    print(df.loc[df["placed_binary"].isna(), "PlacementStatus"].value_counts())
    raise ValueError("Please check the PlacementStatus labels.")

print("\nPlacement distribution after cleaning:")
print(df["PlacementStatus"].value_counts())

print("\nPlacement percentages:")
print(
    (df["PlacementStatus"].value_counts(normalize=True) * 100)
    .round(2)
)

# Compare numeric feature averages by placement status
numeric_cols = df.select_dtypes(include="number").columns.drop(
    "placed_binary"
)

print("\nNumeric feature averages by placement status:")
print(
    df.groupby("PlacementStatus")[numeric_cols]
    .mean()
    .round(2)
    .T
)

# Correlations with the binary target
print("\nCorrelation with placement:")
print(
    df[numeric_cols.tolist() + ["placed_binary"]]
    .corr(numeric_only=True)["placed_binary"]
    .drop("placed_binary")
    .sort_values(key=abs, ascending=False)
    .round(3)
)

print("\nCategorical feature placement rates:")
for col in ["ExtracurricularActivities", "PlacementTraining"]:
    print(f"\n{col}:")
    print(
        df.groupby(col, dropna=False)["placed_binary"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "placement_rate"})
        .round(3)
    )
