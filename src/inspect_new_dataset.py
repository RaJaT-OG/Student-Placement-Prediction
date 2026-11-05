
from pathlib import Path
import pandas as pd

# Project root directory
ROOT = Path(__file__).resolve().parent.parent
file_path = ROOT / "data" / "campus_recruitment.csv"

# Check whether the file exists
if not file_path.exists():
    print(f"ERROR: Dataset not found at {file_path}")
    raise SystemExit(1)

# Load dataset
df = pd.read_csv(file_path)

print("=" * 50)
print("DATASET INSPECTION")
print("=" * 50)

print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nDataset information:")
df.info()

# Check placement target
if "PlacementStatus" in df.columns:
    print("\nPlacement distribution:")
    print(df["PlacementStatus"].value_counts(dropna=False))

    print("\nPlacement percentages:")
    print(
        (df["PlacementStatus"].value_counts(normalize=True) * 100)
        .round(2)
    )
else:
    print("\nWARNING: 'PlacementStatus' not found.")
    print("Check the column names above.")
