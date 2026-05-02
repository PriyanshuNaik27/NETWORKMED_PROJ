"""
STEP 5: Train LD50 Prediction Model
=====================================
LD50 column is already in log10(mg/kg) scale.
  2.265 → 10^2.265 = 184 mg/kg  (4-nitroaniline)
  2.838 → 10^2.838 = 688 mg/kg  (4-nitrophenol)

So we predict log10(LD50) directly — no extra transformation needed.
"""

import numpy as np
import pandas as pd
import os, joblib, json
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def smiles_to_features(smiles: str, n_bits: int = 2048) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = np.array(
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits),
        dtype=np.float32
    )
    desc = np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.RingCount(mol),
    ], dtype=np.float32)
    return np.concatenate([fp, desc])


def train_ld50_model(ld50_csv: str, model_dir: str) -> dict:
    print("\n" + "="*55)
    print("LD50 MODEL: Acute Toxicity Predictor (Regression)")
    print("="*55)

    df = pd.read_csv(ld50_csv)
    df.columns = df.columns.str.strip()

    smiles_col = next((c for c in df.columns if c.lower() == "smiles"), None)
    ld50_col   = next((c for c in df.columns if c.lower() == "ld50"), None)

    if not smiles_col or not ld50_col:
        raise ValueError(f"Need 'smiles' and 'ld50' columns. Found: {df.columns.tolist()}")

    df = df[[smiles_col, ld50_col]].dropna()
    print(f"  Compounds: {len(df)}")
    print(f"  LD50 range (log10): {df[ld50_col].min():.3f} – {df[ld50_col].max():.3f}")
    print(f"  LD50 range (mg/kg): {10**df[ld50_col].min():.1f} – {10**df[ld50_col].max():.1f}")

    print("\nBuilding molecular features...")
    X_list, y_list = [], []
    for _, row in df.iterrows():
        feat = smiles_to_features(str(row[smiles_col]))
        if feat is not None:
            X_list.append(feat)
            y_list.append(float(row[ld50_col]))  # already log10 scale — use directly

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"  Featurized: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=15, n_jobs=-1, random_state=42
        ),
    }

    best_model, best_r2, best_name = None, -999, ""
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        r2  = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        # Convert MAE from log10 to fold-error: 10^MAE
        fold_error = 10 ** mae
        print(f"  {name}: R²={r2:.4f} | MAE={mae:.3f} log10 units (~{fold_error:.1f}x fold error)")
        if r2 > best_r2:
            best_r2, best_model, best_name = r2, m, name

    print(f"\n  Best: {best_name} (R²={best_r2:.4f})")

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_dir, "ld50_model.pkl"))

    metrics = {
        "model_type": best_name,
        "r2": round(best_r2, 4),
        "ld50_unit": "log10(mg/kg)",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "ld50_log10_range": [round(float(y.min()), 3), round(float(y.max()), 3)],
        "ld50_mgkg_range":  [round(float(10**y.min()), 1), round(float(10**y.max()), 1)],
    }
    with open(os.path.join(model_dir, "ld50_model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ LD50 model saved!")
    return metrics


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_ld50_model(
        ld50_csv=os.path.join(BASE, "data", "ld50.csv"),
        model_dir=os.path.join(BASE, "models")
    )
