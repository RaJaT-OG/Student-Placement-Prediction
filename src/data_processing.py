
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Find CSV files
csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    print("No CSV files found in the data folder.")
    print("Please add your dataset to:", DATA_DIR)
else:
    for file_path in csv_files:
        df = pd.read_csv(file_path)

        print(f"\n{'=' * 50}")
        print("File:", file_path.name)
        print("Dataset shape:", df.shape)

        print("\nFirst 5 rows:")
        print(df.head())

        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nData types:")
        print(df.dtypes)

        print("\nMissing values:")
        print(df.isnull().sum()[df.isnull().sum() > 0])

        print("\nDuplicate rows:", df.duplicated().sum())
