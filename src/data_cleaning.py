
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find the training CSV
csv_files = [
    file for file in DATA_DIR.glob("*.csv")
    if "train" in file.stem.lower()
]

if not csv_files:
    raise FileNotFoundError(
        "Training CSV not found. Place your training dataset "
        "inside the data folder and include 'train' in its filename."
    )

input_path = csv_files[0]
df = pd.read_csv(input_path)

print("Loaded:", input_path.name)
print("Original shape:", df.shape)

# 1. Inspect missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 2. Inspect duplicate rows
duplicate_count = df.duplicated().sum()
print("\nDuplicate rows:", duplicate_count)

# 3. Remove exact duplicate rows
df = df.drop_duplicates().copy()

# 4. Identify columns
# Keep the target and identifier unchanged.
target_column = "placed" if "placed" in df.columns else None
id_column = "student_id" if "student_id" in df.columns else None

excluded_columns = [
    column for column in [target_column, id_column]
    if column is not None
]

feature_columns = [
    column for column in df.columns
    if column not in excluded_columns
]

# 5. Fill missing feature values
for column in feature_columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        # Median is robust to extreme values
        median_value = df[column].median()

        if pd.notna(median_value):
            df[column] = df[column].fillna(median_value)
    else:
        # Use the most frequent category
        mode_values = df[column].mode(dropna=True)

        if not mode_values.empty:
            df[column] = df[column].fillna(mode_values.iloc[0])

# 6. Report remaining missing values
print("\nMissing values after cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 7. Save cleaned training data
output_path = OUTPUT_DIR / "student_placement_cleaned.csv"
df.to_csv(output_path, index=False)

print("\nCleaned shape:", df.shape)
print("Saved cleaned dataset to:", output_path)
