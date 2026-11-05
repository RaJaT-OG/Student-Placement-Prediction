
from pathlib import Path
import pandas as pd

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Load datasets
train_path = DATA_DIR / "student_placement_train.csv"
test_path = DATA_DIR / "student_placement_test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("TRAINING DATASET")
print("Shape:", train_df.shape)
print("\nFirst five rows:")
print(train_df.head())

print("\nColumn information:")
train_df.info()

print("\nMissing values:")
print(train_df.isnull().sum().sort_values(ascending=False).head(15))

print("\nDuplicate rows:", train_df.duplicated().sum())

print("\nPlacement distribution:")
print(train_df["placed"].value_counts(dropna=False))

print("\nTEST DATASET")
print("Shape:", test_df.shape)
print(test_df.head())
