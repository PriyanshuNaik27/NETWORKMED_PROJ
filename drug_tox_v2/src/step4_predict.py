"""
STEP 4: Prediction Pipeline
============================
Three-layer toxicity prediction for ANY new SMILES:

  Layer 1 — ML Tox21 model    : trained on Tox21 (endocrine/cancer toxicity)
  Layer 2 — LD50 model        : trained on LD50 data, predicts acute toxicity
                                 for ANY SMILES even if not in the dataset
  Layer 3 — Structural alerts : rule-based, catches cyanide/arsenic/etc instantly

Final score = weighted combination of all available layers.
"""

import numpy as np
import pandas as pd
import joblib
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ── Risk thresholds ───────────────────────────────────────────────────────────
RISK_LEVELS = {
    (0.00, 0.30): ("🟢 LOW",      "Low toxicity risk. Compound appears structurally safe."),
    (0.30, 0.55): ("🟡 MODERATE", "Moderate toxicity risk. Further wet-lab validation recommended."),
    (0.55, 0.75): ("🟠 HIGH",     "High toxicity risk. Significant structural alerts detected."),
    (0.75, 1.01): ("🔴 VERY HIGH","Very high toxicity risk. Strong similarity to known toxins."),
}

# ── Structural alerts ─────────────────────────────────────────────────────────
STRUCTURAL_ALERTS = {
    "Cyanide":           "[C-]#N",
    "Nitrile":           "C#N",
    "Azide":             "N=[N+]=[N-]",
    "Arsenic":           "[As]",
    "Mercury":           "[Hg]",
    "Lead":              "[Pb]",
    "Cadmium":           "[Cd]",
    "Phosgene":          "ClC(=O)Cl",
    "Mustard gas":       "ClCCSCCCl",
    "Epoxide":           "C1OC1",
    "Peroxide":          "OO",
    "Nitro group":       "[N+](=O)[O-]",
    "Acyl halide":       "C(=O)Cl",
    "Isocyanate":        "N=C=O",
}


# ── Feature builders ──────────────────────────────────────────────────────────

def smiles_to_morgan_fp(smiles: str, n_bits: int = 2048) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def smiles_to_ld50_features(smiles: str, n_bits: int = 2048) -> np.ndarray | None:
    """
    Morgan FP + 10 physicochemical descriptors.
    Same feature set used when training the LD50 model.
    """
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


def get_mol_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "molecular_weight":  round(Descriptors.MolWt(mol), 2),
        "logP":              round(Descriptors.MolLogP(mol), 3),
        "H_bond_donors":     Descriptors.NumHDonors(mol),
        "H_bond_acceptors":  Descriptors.NumHAcceptors(mol),
        "TPSA":              round(Descriptors.TPSA(mol), 2),
        "rotatable_bonds":   Descriptors.NumRotatableBonds(mol),
        "aromatic_rings":    Descriptors.NumAromaticRings(mol),
    }


def get_lipinski_summary(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    mw, logp = Descriptors.MolWt(mol), Descriptors.MolLogP(mol)
    hbd, hba = Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)
    violations = (
        (["MW>500"] if mw > 500 else []) +
        (["LogP>5"]  if logp > 5 else []) +
        (["HBD>5"]   if hbd > 5 else []) +
        (["HBA>10"]  if hba > 10 else [])
    )
    n = len(violations)
    return {
        "molecular_weight": round(mw, 2), "logP": round(logp, 3),
        "H_bond_donors": hbd, "H_bond_acceptors": hba,
        "violations_count": n, "violations": violations,
        "status": ["Excellent","Acceptable","Poor"][min(n,2)],
        "interpretation": [
            "Fully satisfies Lipinski Rule of 5.",
            "One Lipinski violation. May still have reasonable oral bioavailability.",
            "Multiple Lipinski violations. Oral drug-likeness may be weak."
        ][min(n,2)],
    }


def check_structural_alerts(smiles: str) -> list:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    found = []
    for name, smarts in STRUCTURAL_ALERTS.items():
        try:
            patt = Chem.MolFromSmarts(smarts)
            if patt and mol.HasSubstructMatch(patt):
                found.append(name)
        except Exception:
            pass
    return found


def ld50_value_to_score(ld50_mg_per_kg: float) -> float:
    """
    Convert predicted LD50 (mg/kg) to 0–1 toxicity score.
    GHS acute toxicity categories:
      ≤ 5      → 1.00  extremely toxic
      ≤ 50     → 0.90  very toxic
      ≤ 300    → 0.70  toxic
      ≤ 2000   → 0.40  harmful
      ≤ 5000   → 0.20  slightly harmful
      > 5000   → 0.05  practically non-toxic
    """
    if ld50_mg_per_kg <= 5:    return 1.00
    if ld50_mg_per_kg <= 50:   return 0.90
    if ld50_mg_per_kg <= 300:  return 0.70
    if ld50_mg_per_kg <= 2000: return 0.40
    if ld50_mg_per_kg <= 5000: return 0.20
    return 0.05


# ── Main predictor ────────────────────────────────────────────────────────────

class ToxicityPredictor:

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
            )
        self.model_dir  = model_dir
        self.ld50_model = None
        self._load_models()

    def _load_models(self):
        # Tox21 models (required)
        prom_path = os.path.join(self.model_dir, "promiscuity_model.pkl")
        tox_path  = os.path.join(self.model_dir, "toxicity_model.pkl")
        if not os.path.exists(prom_path) or not os.path.exists(tox_path):
            raise FileNotFoundError("Tox21 models not found. Run run_pipeline.py first.")
        self.prom_model = joblib.load(prom_path)
        self.tox_model  = joblib.load(tox_path)
        print("✅ Tox21 models loaded.")

        # LD50 model (optional but strongly recommended)
        ld50_model_path = os.path.join(self.model_dir, "ld50_model.pkl")
        if os.path.exists(ld50_model_path):
            self.ld50_model = joblib.load(ld50_model_path)
            print("✅ LD50 prediction model loaded.")
        else:
            print("ℹ️  No LD50 model found.")
            print("   → Place ld50.csv in data/ and run:")
            print("     python -c \"from src.step5_train_ld50_model import train_ld50_model; "
                  "train_ld50_model('data/ld50.csv', 'models')\"")

    def _predict_ld50(self, smiles: str):
        """
        Predict LD50 for ANY SMILES using the trained LD50 model.
        Returns (predicted_ld50_mg_per_kg, toxicity_score_0_to_1) or (None, None).
        """
        if self.ld50_model is None:
            return None, None
        feat = smiles_to_ld50_features(smiles)
        if feat is None:
            return None, None
        # Model predicts log10(LD50 in mg/kg) — convert back with 10^x
        log10_pred = float(self.ld50_model.predict([feat])[0])
        ld50_val   = max(0.1, float(10 ** log10_pred))
        score    = ld50_value_to_score(ld50_val)
        return round(ld50_val, 2), round(score, 4)

    def predict(self, smiles: str, drug_name: str = None) -> dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Invalid SMILES: '{smiles}'"}

        # ── Layer 1: Tox21 ML model ───────────────────────────────────────────
        fp        = smiles_to_morgan_fp(smiles)
        pred_prom = max(0, round(float(self.prom_model.predict([fp])[0]), 3))
        features  = np.append(fp, pred_prom).reshape(1, -1)
        ml_score  = round(float(self.tox_model.predict_proba(features)[0][1]), 4)

        # ── Layer 2: LD50 model (predicts for ANY new SMILES) ─────────────────
        ld50_val, ld50_score = self._predict_ld50(smiles)

        # ── Layer 3: Structural alerts ────────────────────────────────────────
        alerts = check_structural_alerts(smiles)

        # ── Combine scores ────────────────────────────────────────────────────
        if alerts:
            # Structural alert always wins — force very high risk
            final_score = max(0.90, ml_score)
            method = "Structural alert override"
        elif ld50_score is not None:
            # LD50 model predicts acute toxicity better than Tox21
            # Weight: ML=35%, LD50=65%
            final_score = round(0.35 * ml_score + 0.65 * ld50_score, 4)
            method = "ML (35%) + LD50 model (65%)"
        else:
            # Fallback: ML only
            final_score = ml_score
            method = "ML model only (add ld50.csv for better predictions)"

        # ── Risk label ────────────────────────────────────────────────────────
        risk_label, interpretation = "UNKNOWN", ""
        for (lo, hi), (label, desc) in RISK_LEVELS.items():
            if lo <= final_score < hi:
                risk_label, interpretation = label, desc
                break

        if alerts:
            interpretation = (
                f"⚠️ Known toxic substructure(s) detected: {', '.join(alerts)}. "
                f"Risk forced to VERY HIGH regardless of ML score."
            )

        # ── LD50 category label ───────────────────────────────────────────────
        ld50_category = None
        if ld50_val is not None:
            if ld50_val <= 5:      ld50_category = "Extremely toxic (GHS Cat 1)"
            elif ld50_val <= 50:   ld50_category = "Very toxic (GHS Cat 2)"
            elif ld50_val <= 300:  ld50_category = "Toxic (GHS Cat 3)"
            elif ld50_val <= 2000: ld50_category = "Harmful (GHS Cat 4)"
            elif ld50_val <= 5000: ld50_category = "Slightly harmful (GHS Cat 5)"
            else:                  ld50_category = "Practically non-toxic"

        return {
            "smiles":               smiles,
            "drug_name":            drug_name,
            "toxicity_score":       round(final_score, 4),
            "toxicity_percent":     f"{final_score * 100:.1f}%",
            "risk_level":           risk_label,
            "interpretation":       interpretation,
            "predicted_promiscuity": pred_prom,
            "score_breakdown": {
                "ml_model_score":        ml_score,
                "ld50_predicted_mg_per_kg": ld50_val,
                "ld50_score":            ld50_score,
                "ld50_category":         ld50_category,
                "structural_alerts":     alerts,
                "scoring_method":        method,
            },
            "molecular_properties": get_mol_descriptors(smiles),
            "lipinski":             get_lipinski_summary(smiles),
        }

    def predict_batch(self, smiles_list: list) -> list:
        return [self.predict(s) for s in smiles_list]

    def predict_from_name(self, drug_name: str) -> dict:
        import requests
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{requests.utils.quote(drug_name)}/property/IsomericSMILES/JSON"
        )
        try:
            r      = requests.get(url, timeout=10)
            props  = r.json()["PropertyTable"]["Properties"][0]
            smiles = props.get("IsomericSMILES") or props.get("SMILES")
            return self.predict(smiles, drug_name=drug_name)
        except Exception as e:
            return {"error": f"Could not fetch SMILES for '{drug_name}': {e}"}


if __name__ == "__main__":
    predictor = ToxicityPredictor()

    tests = [
        ("Aspirin",     "CC(=O)Oc1ccccc1C(=O)O"),
        ("Cyanide",     "[C-]#N"),
        ("Naphthalene", "c1ccc2ccccc2c1"),
        ("Paracetamol", "CC(=O)Nc1ccc(O)cc1"),
        ("Ethanol",     "CCO"),
        ("Mustard gas", "ClCCSCCCl"),
        ("Ibuprofen",   "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ]

    print("\n" + "="*80)
    print(f"{'Drug':<14} {'Final':>7} {'ML':>7} {'LD50(mg/kg)':>12} {'LD50 Score':>10} {'Alerts':<15} {'Risk'}")
    print("="*80)
    for name, smi in tests:
        r  = predictor.predict(smi, drug_name=name)
        bd = r["score_breakdown"]
        al = ", ".join(bd["structural_alerts"]) if bd["structural_alerts"] else "None"
        ld = f"{bd['ld50_predicted_mg_per_kg']:.1f}" if bd['ld50_predicted_mg_per_kg'] else "N/A"
        ls = f"{bd['ld50_score']:.4f}" if bd['ld50_score'] else "N/A"
        print(f"{name:<14} {r['toxicity_score']:>7.4f} {bd['ml_model_score']:>7.4f} {ld:>12} {ls:>10} {al:<15} {r['risk_level']}")
