
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the cleaned dataset if available, otherwise load training data
cleaned_path = OUTPUT_DIR / "student_placement_cleaned.csv"
raw_path = DATA_DIR / "student_placement_train.csv"

if cleaned_path.exists():
    df = pd.read_csv(cleaned_path)
elif raw_path.exists():
    df = pd.read_csv(raw_path)
else:
    raise FileNotFoundError(
        "Training dataset not found. Check the data folder."
    )

TARGET = "placed"
ID_COLUMN = "student_id"

if TARGET not in df.columns:
    raise ValueError("The dataset must contain the 'placed' target column.")

# Remove rows with missing target labels
df = df.dropna(subset=[TARGET]).copy()

# Separate target, identifier and predictive features
y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET])

if ID_COLUMN in X.columns:
    X = X.drop(columns=[ID_COLUMN])

# Remove completely empty feature columns
empty_columns = X.columns[X.isna().all()].tolist()
if empty_columns:
    print("Removing completely empty columns:", empty_columns)
    X = X.drop(columns=empty_columns)

# Identify feature types
numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Dataset shape:", df.shape)
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# Split before fitting preprocessing steps
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine transformations
transformers = []

if numeric_features:
    transformers.append(("numeric", numeric_pipeline, numeric_features))

if categorical_features:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(transformers=transformers)

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_valid_processed = preprocessor.transform(X_valid)

# Convert sparse matrices to dense arrays when necessary
if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()

if hasattr(X_valid_processed, "toarray"):
    X_valid_processed = X_valid_processed.toarray()

# Select useful features using mutual information
k = min(20, X_train_processed.shape[1])

selector = SelectKBest(
    score_func=mutual_info_classif,
    k=k
)

X_train_selected = selector.fit_transform(X_train_processed, y_train)
X_valid_selected = selector.transform(X_valid_processed)

# Save processed training and validation data
train_output = pd.DataFrame(X_train_selected)
train_output["placed"] = y_train.reset_index(drop=True)

valid_output = pd.DataFrame(X_valid_selected)
valid_output["placed"] = y_valid.reset_index(drop=True)

train_output.to_csv(
    OUTPUT_DIR / "placement_train_processed.csv",
    index=False
)
valid_output.to_csv(
    OUTPUT_DIR / "placement_validation_processed.csv",
    index=False
)

print("\nPreprocessing complete.")
print("Training features:", X_train_selected.shape)
print("Validation features:", X_valid_selected.shape)
print("Selected features:", k)
print("Files saved in:", OUTPUT_DIR)
