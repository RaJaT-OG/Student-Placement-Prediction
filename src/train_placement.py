"""Audit placement datasets, compare models, evaluate, and save pipelines.

Run from the project root with the virtual environment's Python executable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
TARGETS = {
    "student_placement_train.csv": "placed",
    "campus_recruitment.csv": "PlacementStatus",
}


def encode_target(values: pd.Series, dataset_name: str) -> tuple[pd.Series, dict[str, int]]:
    if dataset_name == "campus_recruitment.csv":
        normalized = values.astype("string").str.strip().str.lower()
        mapping = {"notplaced": 0, "not placed": 0, "placed": 1}
        encoded = normalized.map(mapping)
        if encoded.isna().any():
            raise ValueError(f"Unexpected PlacementStatus values: {sorted(normalized[encoded.isna()].dropna().unique())}")
        return encoded.astype(int), mapping

    numeric = pd.to_numeric(values, errors="coerce")
    if set(numeric.dropna().unique().tolist()) <= {0, 1}:
        return numeric.astype(int), {"0": 0, "1": 1}
    normalized = values.astype("string").str.strip().str.lower()
    mapping = {"not placed": 0, "notplaced": 0, "no": 0, "false": 0,
               "placed": 1, "yes": 1, "true": 1}
    encoded = normalized.map(mapping)
    if encoded.isna().any():
        raise ValueError("Target must contain binary 0/1 or recognizable placement labels.")
    return encoded.astype(int), mapping


def make_candidates(X: pd.DataFrame) -> dict[str, Pipeline]:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]), categorical))
    if not transformers:
        raise ValueError("No usable feature columns remain after excluding target and ID.")

    preprocess = ColumnTransformer(transformers, remainder="drop")
    return {
        "Logistic Regression": Pipeline([
            ("preprocess", clone(preprocess)),
            ("model", LogisticRegression(C=0.25, max_iter=3000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", clone(preprocess)),
            ("model", RandomForestClassifier(
                n_estimators=400, max_features="sqrt", min_samples_leaf=3,
                class_weight="balanced_subsample", n_jobs=-1, random_state=42,
            )),
        ]),
        "Extra Trees": Pipeline([
            ("preprocess", clone(preprocess)),
            ("model", ExtraTreesClassifier(
                n_estimators=500, max_features="sqrt", min_samples_leaf=3,
                class_weight="balanced", n_jobs=-1, random_state=42,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocess", clone(preprocess)),
            ("model", GradientBoostingClassifier(
                n_estimators=250, learning_rate=0.04, max_depth=2,
                min_samples_leaf=8, random_state=42,
            )),
        ]),
    }


def audit_dataset(filename: str):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    raw = pd.read_csv(path)
    target = TARGETS[filename]
    if target not in raw:
        raise ValueError(f"Expected target column {target!r} in {filename}.")

    missing_target = int(raw[target].isna().sum())
    deduplicated = raw.drop_duplicates()
    labeled = deduplicated.dropna(subset=[target]).copy()
    y, mapping = encode_target(labeled[target], filename)
    id_columns = [c for c in labeled.columns if c.lower() in {"student_id", "id", "record_id"} or c.lower().endswith("_id")]
    X = labeled.drop(columns=[target, *id_columns], errors="ignore")
    all_missing = X.columns[X.isna().all()].tolist()
    X = X.drop(columns=all_missing).reset_index(drop=True)
    y = y.reset_index(drop=True)
    feature_keys = X.astype("string").fillna("<MISSING>").agg("\x1f".join, axis=1)
    feature_labels = pd.DataFrame({"key": feature_keys, "label": y}).groupby("key")["label"].nunique()
    conflicting_keys = feature_labels[feature_labels > 1].index
    conflicting_rows = feature_keys.isin(conflicting_keys)
    conflicting_rows_removed = int(conflicting_rows.sum())
    X = X.loc[~conflicting_rows].reset_index(drop=True)
    y = y.loc[~conflicting_rows].reset_index(drop=True)
    duplicate_features = int(X.astype("string").fillna("<MISSING>").duplicated().sum())
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    report = {
        "dataset": filename,
        "raw_rows": int(len(raw)),
        "labeled_rows_after_dedup": int(len(labeled)),
        "duplicate_rows_removed": int(len(raw) - len(deduplicated)),
        "missing_target_rows_removed": missing_target,
        "feature_count": int(X.shape[1]),
        "numeric_feature_count": len(numeric),
        "categorical_feature_count": len(categorical),
        "missing_feature_cells": int(X.isna().sum().sum()),
        "features_with_missing_values": X.columns[X.isna().any()].tolist(),
        "all_missing_features_removed": all_missing,
        "id_columns_excluded": id_columns,
        "duplicate_feature_rows": duplicate_features,
        "conflicting_duplicate_feature_sets": int(len(conflicting_keys)),
        "conflicting_feature_rows_removed": conflicting_rows_removed,
        "strongest_numeric_target_correlations": (
            X[numeric].corrwith(y).abs().sort_values(ascending=False).head(5).to_dict()
            if numeric else {}
        ),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "positive_rate": float(y.mean()),
        "target_mapping": mapping,
    }
    return raw, X, y, target, report


def train_dataset(filename: str) -> dict:
    raw, X, y, target, audit = audit_dataset(filename)
    print(f"\n{'=' * 78}\nDATASET: {filename}")
    print(f"Rows: {audit['raw_rows']:,} raw -> {len(X):,} labeled unique | features: {X.shape[1]} ({audit['numeric_feature_count']} numeric, {audit['categorical_feature_count']} categorical)")
    print(f"Target counts: {audit['class_counts']} | positive rate: {audit['positive_rate']:.1%}")
    print(f"Missing feature cells: {audit['missing_feature_cells']:,} | excluded IDs: {audit['id_columns_excluded'] or 'none'}")
    print(f"Duplicate feature rows: {audit['duplicate_feature_rows']:,}")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    candidates = make_candidates(X_train)
    rows = []
    print("\nThree-fold ROC-AUC cross-validation on training partition:", flush=True)
    for name, pipeline in candidates.items():
        scores = cross_val_score(clone(pipeline), X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        rows.append({"Model": name, "CV_ROC_AUC_mean": scores.mean(), "CV_ROC_AUC_std": scores.std()})
        print(f"  {name:22s} {scores.mean():.4f} +/- {scores.std():.4f}", flush=True)

    comparison = pd.DataFrame(rows).sort_values("CV_ROC_AUC_mean", ascending=False).reset_index(drop=True)
    best_name = str(comparison.iloc[0]["Model"])
    best_pipeline = clone(candidates[best_name]).fit(X_train, y_train)
    probabilities = best_pipeline.predict_proba(X_valid)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    holdout = {
        "accuracy": float(accuracy_score(y_valid, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_valid, predictions)),
        "precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "f1": float(f1_score(y_valid, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_valid, predictions, labels=[0, 1]).tolist(),
        "majority_class_accuracy_baseline": float(max(y_train.mean(), 1 - y_train.mean())),
        "validation_rows": int(len(y_valid)),
        "selected_threshold": 0.5,
    }
    for name, pipeline in candidates.items():
        fitted = clone(pipeline).fit(X_train, y_train)
        p = fitted.predict_proba(X_valid)[:, 1]
        pred = (p >= 0.5).astype(int)
        comparison.loc[comparison.Model == name, "Holdout_Accuracy"] = accuracy_score(y_valid, pred)
        comparison.loc[comparison.Model == name, "Holdout_ROC_AUC"] = roc_auc_score(y_valid, p)
        comparison.loc[comparison.Model == name, "Holdout_F1"] = f1_score(y_valid, pred, zero_division=0)

    print(f"\nSelected by 3-fold CV ROC-AUC: {best_name}")
    print("Untouched 20% validation metrics:")
    for key, value in holdout.items():
        print(f"  {key}: {value}")
    print("\nClassification report at threshold 0.50:")
    print(classification_report(y_valid, predictions, target_names=["Not placed", "Placed"], zero_division=0))

    stem = "student_placement" if filename == "student_placement_train.csv" else "campus_recruitment"
    comparison.to_csv(OUTPUT_DIR / f"{stem}_model_comparison.csv", index=False)
    if holdout["roc_auc"] >= 0.60:
        final_pipeline = clone(candidates[best_name]).fit(X, y)
        artifact = {
            "pipeline": final_pipeline,
            "target": target,
            "features": X.columns.tolist(),
            "excluded_columns": audit["id_columns_excluded"],
            "class_mapping": audit["target_mapping"],
            "selected_model": best_name,
            "holdout_metrics": holdout,
            "dataset": filename,
        }
        joblib.dump(artifact, MODEL_DIR / f"{stem}_pipeline.joblib")

    if filename == "student_placement_train.csv" and holdout["roc_auc"] >= 0.60:
        test_path = DATA_DIR / "student_placement_test.csv"
        if test_path.exists():
            test = pd.read_csv(test_path)
            ids = test["student_id"].copy() if "student_id" in test else pd.Series(np.arange(len(test)))
            test_probability = final_pipeline.predict_proba(test.reindex(columns=X.columns))[:, 1]
            pd.DataFrame({"student_id": ids, "placed": test_probability}).to_csv(
                OUTPUT_DIR / "student_placement_test_predictions.csv", index=False,
            )
            print(f"Scored unlabeled test rows: {len(test):,}; probability file saved.")

    report = {
        **audit,
        "selected_model": best_name,
        "holdout": holdout,
        "cross_validation": comparison[["Model", "CV_ROC_AUC_mean", "CV_ROC_AUC_std"]].to_dict(orient="records"),
        "validation_classification_report": classification_report(
            y_valid, predictions, target_names=["Not placed", "Placed"], output_dict=True, zero_division=0,
        ),
    }
    (OUTPUT_DIR / f"{stem}_training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if holdout["roc_auc"] < 0.60:
        print("WARNING: ROC-AUC is near chance. This dataset shows no reliable signal; majority-class accuracy is misleading. No deployable model or test predictions were saved.")
    elif holdout["roc_auc"] >= 0.85:
        print("Strong validation ranking signal: ROC-AUC is at least 0.85.")
    return {"dataset": filename, "model": best_name, **holdout}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*TARGETS, "all"], default="all",
                        help="Which labeled dataset(s) to audit and train (default: all).")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    filenames = list(TARGETS) if args.dataset == "all" else [args.dataset]
    results = [train_dataset(name) for name in filenames]
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "training_summary.csv", index=False)
    print(f"\nSaved run summary: {OUTPUT_DIR / 'training_summary.csv'}")


if __name__ == "__main__":
    main()
