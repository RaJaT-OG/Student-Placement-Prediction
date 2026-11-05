
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "student_placement_train.csv")

df = df.dropna(subset=["placed"]).copy()
y = df["placed"].astype(int)

X = df.drop(columns=["placed", "student_id"], errors="ignore")
X = X.dropna(axis=1, how="all")

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

transformers = []

if numeric_cols:
    transformers.append((
        "numeric",
        SimpleImputer(strategy="median"),
        numeric_cols
    ))

if categorical_cols:
    transformers.append((
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_cols
    ))

preprocessor = ColumnTransformer(transformers=transformers)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train_transformed = preprocessor.fit_transform(X_train)

scores = mutual_info_classif(
    X_train_transformed,
    y_train,
    random_state=42
)

feature_names = preprocessor.get_feature_names_out()

importance = pd.DataFrame({
    "feature": feature_names,
    "mutual_information": scores
}).sort_values("mutual_information", ascending=False)

print("\nTop 20 features by mutual information:")
print(importance.head(20).to_string(index=False))

print("\nTarget distribution by class:")
print(y_train.value_counts(normalize=True).sort_index())

print("\nNumeric feature averages by placement status:")
for col in numeric_cols:
    print(f"\n{col}")
    print(X_train.groupby(y_train)[col].mean().round(3))
