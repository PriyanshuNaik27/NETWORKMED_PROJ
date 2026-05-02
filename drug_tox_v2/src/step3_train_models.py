"""
STEP 3: Train Models
=====================
Model 1: SMILES fingerprint → promiscuity score  (Regression)
Model 2: SMILES fingerprint + promiscuity score → toxicity  (Classification)
"""

import numpy as np
import pandas as pd
import os
import joblib
import json

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, r2_score,
    roc_auc_score, accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler


def train_promiscuity_model(X: np.ndarray, y: np.ndarray, model_dir: str) -> dict:
    """
    Train Model 1: fingerprint → promiscuity score.
    Returns evaluation metrics.
    """
    print("\n" + "="*50)
    print("MODEL 1: Promiscuity Predictor (Regression)")
    print("="*50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "promiscuity_model.pkl"))

    metrics = {
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "cv_r2_mean": round(cv_scores.mean(), 4),
        "cv_r2_std": round(cv_scores.std(), 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    with open(os.path.join(model_dir, "prom_model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Promiscuity model saved!")
    return metrics


def train_toxicity_model(X: np.ndarray, y: np.ndarray, model_dir: str) -> dict:
    """
    Train Model 2: fingerprint + promiscuity → toxicity score.
    Returns evaluation metrics.
    """
    print("\n" + "="*50)
    print("MODEL 2: Toxicity Predictor (Classification)")
    print("="*50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Try both RF and GBM, pick better one
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=15,
            class_weight='balanced', n_jobs=-1, random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05,
            max_depth=5, random_state=42
        )
    }

    best_model, best_auc, best_name = None, 0, ""

    for name, m in models.items():
        m.fit(X_train, y_train)
        proba = m.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        print(f"  {name}: AUC = {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_model = m
            best_name = name

    print(f"\n  Best model: {best_name} (AUC={best_auc:.4f})")

    # Full evaluation on best model
    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['Non-toxic','Toxic'])}")

    # Save
    joblib.dump(best_model, os.path.join(model_dir, "toxicity_model.pkl"))

    metrics = {
        "model_type": best_name,
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "toxic_in_test": int(y_test.sum()),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    with open(os.path.join(model_dir, "tox_model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Toxicity model saved!")
    return metrics


def run_training(features_dir: str, model_dir: str):
    print("Loading feature matrices...")
    X_prom = np.load(os.path.join(features_dir, "X_prom.npy"))
    y_prom = np.load(os.path.join(features_dir, "y_prom.npy"))
    X_tox  = np.load(os.path.join(features_dir, "X_tox.npy"))
    y_tox  = np.load(os.path.join(features_dir, "y_tox.npy"))

    print(f"  Promiscuity data: {X_prom.shape} → {y_prom.shape}")
    print(f"  Toxicity data:    {X_tox.shape} → {y_tox.shape}")

    prom_metrics = train_promiscuity_model(X_prom, y_prom, model_dir)
    tox_metrics  = train_toxicity_model(X_tox, y_tox, model_dir)

    # Save combined summary
    summary = {"promiscuity_model": prom_metrics, "toxicity_model": tox_metrics}
    with open(os.path.join(model_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"  Promiscuity R²  : {prom_metrics['r2']}")
    print(f"  Toxicity ROC-AUC: {tox_metrics['roc_auc']}")


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_training(
        features_dir=os.path.join(BASE, "data", "features"),
        model_dir=os.path.join(BASE, "models")
    )
