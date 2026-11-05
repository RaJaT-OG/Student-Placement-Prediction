
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned data
file_path = DATA_DIR / "student_placement_train.csv"

if not file_path.exists():
    file_path = OUTPUT_DIR / "student_placement_cleaned.csv"

if not file_path.exists():
    raise FileNotFoundError(
        "Dataset not found. Place the training CSV in data/ "
        "or run data_cleaning.py first."
    )

df = pd.read_csv(file_path)
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Target column
target = "placed"

if target not in df.columns:
    raise ValueError("The dataset must contain a 'placed' target column.")

# 1. Placement distribution
print("\nPlacement counts:")
print(df[target].value_counts(dropna=False))

print("\nPlacement percentages:")
print((df[target].value_counts(normalize=True) * 100).round(2))

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x=target)
plt.title("Placement Status Distribution")
plt.xlabel("Placement (0 = Not Placed, 1 = Placed)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "placement_distribution.png", dpi=150)
plt.show()

# 2. Numerical feature distributions
numeric_columns = df.select_dtypes(include="number").columns.tolist()
numeric_features = [
    col for col in numeric_columns
    if col not in ["placed", "student_id"]
]

if numeric_features:
    df[numeric_features].hist(
        figsize=(14, 10),
        bins=20,
        edgecolor="black"
    )
    plt.suptitle("Numerical Feature Distributions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "numerical_distributions.png", dpi=150)
    plt.show()

# 3. Correlation heatmap
if len(numeric_features) >= 2:
    plt.figure(figsize=(12, 8))
    correlation = df[numeric_features + [target]].corr(numeric_only=True)
    sns.heatmap(correlation, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.show()

# 4. Outlier inspection using boxplots
if numeric_features:
    columns_to_plot = numeric_features[:12]
    df[columns_to_plot].plot(
        kind="box",
        subplots=True,
        layout=(-1, 3),
        figsize=(14, 10),
        sharex=False,
        sharey=False
    )
    plt.suptitle("Outlier Inspection Using Boxplots")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "outlier_boxplots.png", dpi=150)
    plt.show()

# 5. IQR outlier counts (inspection only; no rows removed)
print("\nPotential outliers using the IQR rule:")

for column in numeric_features:
    values = df[column].dropna()

    if values.empty:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    count = ((values < lower_bound) | (values > upper_bound)).sum()

    print(f"{column}: {count}")

print("\nEDA complete. Charts saved in:", OUTPUT_DIR)
