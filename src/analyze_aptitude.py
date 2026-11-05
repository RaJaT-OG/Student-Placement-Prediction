
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "campus_recruitment.csv")

df = df.drop_duplicates().copy()

df["PlacementStatus"] = (
    df["PlacementStatus"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"placed": 1, "notplaced": 0})
)

if df["PlacementStatus"].isna().any():
    raise ValueError("Unexpected placement labels.")

print("Aptitude score summary:")
print(df["AptitudeTestScore"].describe().round(2))

# Divide aptitude scores into five groups
df["AptitudeGroup"] = pd.qcut(
    df["AptitudeTestScore"],
    q=5,
    duplicates="drop"
)

summary = df.groupby(
    "AptitudeGroup", observed=True
)["PlacementStatus"].agg(
    Students="count",
    PlacementRate="mean"
)

summary["PlacementRate"] = (
    summary["PlacementRate"] * 100
).round(2)

print("\nPlacement rate by aptitude score:")
print(summary.to_string())

print("\nPlacement rate by training status:")
print(
    df.groupby("PlacementTraining")["PlacementStatus"]
    .agg(["count", "mean"])
    .assign(
        PlacementRate=lambda x: (x["mean"] * 100).round(2)
    )
    .drop(columns="mean")
    .to_string()
)
