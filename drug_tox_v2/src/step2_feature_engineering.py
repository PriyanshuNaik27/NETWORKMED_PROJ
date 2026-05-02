"""
STEP 2: Feature Engineering
============================
- SMILES → Morgan Fingerprints (2048 bits)
- Merge promiscuity scores with Tox21 toxicity labels
- Output: final feature matrix ready for training
"""

import numpy as np
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # suppress RDKit warnings


# ── Fingerprint utilities ────────────────────────────────────────────────────

def smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    """Convert SMILES to Morgan (circular) fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def smiles_to_descriptors(smiles: str) -> dict | None:
    """Additional RDKit molecular descriptors (optional enrichment)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "mol_weight":    Descriptors.MolWt(mol),
        "logp":          Descriptors.MolLogP(mol),
        "hbd":           Descriptors.NumHDonors(mol),
        "hba":           Descriptors.NumHAcceptors(mol),
        "tpsa":          Descriptors.TPSA(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "aromatic_rings":  Descriptors.NumAromaticRings(mol),
    }


# ── Tox21 loader (via DeepChem or fallback CSV) ──────────────────────────────

def load_tox21_labels() -> pd.DataFrame:
    """
    Load Tox21 dataset.
    Priority:
      1. data/tox21.csv  (manual download - recommended)
      2. DeepChem auto-download
      3. 20-compound fallback (too small - avoid)
    Download URL: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz
    """
    # Option 1: Manual CSV
    manual_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "tox21.csv"
    )
    if os.path.exists(manual_path):
        print("Loading Tox21 from local CSV...")
        df = pd.read_csv(manual_path)
        df.columns = df.columns.str.strip()

        # Tox21 exact format:
        #   NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD,
        #   NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53, mol_id, smiles
        # smiles is the LAST column; mol_id is second-to-last

        smiles_col = next((c for c in df.columns if c.lower() == "smiles"), None)
        if smiles_col is None:
            raise ValueError("No smiles column found in tox21.csv")

        skip = {smiles_col.lower(), "mol_id"}
        task_cols = [c for c in df.columns if c.lower() not in skip]
        print(f"  Found {len(task_cols)} toxicity tasks")

        # Aggregate all 12 tasks: compound is toxic if ANY task == 1
        records = []
        for _, row in df.iterrows():
            smi = row[smiles_col]
            if not isinstance(smi, str) or not smi.strip():
                continue
            known = [row[t] for t in task_cols if not pd.isna(row[t])]
            if not known:
                continue
            label = int(any(v == 1 for v in known))
            records.append({"smiles": smi.strip(), "toxicity": label})

        out = pd.DataFrame(records).drop_duplicates(subset="smiles")
        print(f"  Tox21 loaded: {len(out)} compounds | Toxic: {int(out.toxicity.sum())} | Non-toxic: {int((out.toxicity==0).sum())}")
        return out

    # Option 2: DeepChem
    try:
        import deepchem as dc
        print("Loading Tox21 via DeepChem...")
        tasks, datasets, _ = dc.molnet.load_tox21(featurizer='Raw')
        train, valid, test = datasets
        all_data = []
        for ds in [train, valid, test]:
            for smiles, y in zip(ds.ids, ds.y):
                label = y[0]
                if not np.isnan(label):
                    all_data.append({"smiles": smiles, "toxicity": int(label)})
        df = pd.DataFrame(all_data)
        print(f"  Tox21 loaded via DeepChem: {len(df)} compounds")
        return df
    except Exception as e:
        print(f"  DeepChem failed: {e}")
        print("  WARNING: Using 20-sample fallback. Download tox21.csv for real training!")
        print("  URL: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz")
        return _sample_tox21()


def _sample_tox21() -> pd.DataFrame:
    """Fallback: small representative sample with known toxicity labels."""
    data = [
        # (SMILES, toxicity)  1=toxic, 0=non-toxic
        ("CC(=O)Oc1ccccc1C(=O)O", 0),   # Aspirin
        ("c1ccc2ccccc2c1", 1),            # Naphthalene
        ("CCO", 0),                        # Ethanol
        ("c1ccc(cc1)N", 1),               # Aniline
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", 0), # Ibuprofen
        ("O=C(O)c1ccccc1O", 0),           # Salicylic acid
        ("Clc1ccccc1", 1),                # Chlorobenzene
        ("CC(=O)Nc1ccc(O)cc1", 0),        # Paracetamol
        ("c1ccc(cc1)Cl", 1),              # Chlorobenzene isomer
        ("OC(=O)c1ccc(N)cc1", 1),         # 4-aminobenzoic acid
        ("CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", 1),  # Testosterone
        ("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", 0), # Glucose
        ("CCCCc1ccc(cc1)C(=O)O", 0),
        ("Cc1ccc(cc1)S(=O)(=O)N", 1),
        ("c1ccc2[nH]cccc2c1", 1),         # Indole
        ("CC(C)(C)c1ccc(cc1)O", 0),
        ("O=C(O)CCc1ccccc1", 0),
        ("Nc1ccc(cc1)C(=O)O", 1),
        ("CC(O)=O", 0),                    # Acetic acid
        ("CCCCC(=O)O", 0),
    ]
    return pd.DataFrame(data, columns=["smiles", "toxicity"])


# ── Main feature builder ─────────────────────────────────────────────────────

def build_feature_matrix(prom_csv: str, output_dir: str):
    """
    Merge promiscuity data + Tox21 labels → final feature matrix.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load promiscuity data
    prom_df = pd.read_csv(prom_csv)
    prom_df.columns = prom_df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Detect columns
    score_col = next((c for c in prom_df.columns if 'promis' in c or 'index' in c or 'score' in c), prom_df.columns[1])
    smiles_col = 'smiles'
    print(f"Promiscuity column: '{score_col}'")
    print(f"Promiscuity data shape: {prom_df.shape}")

    # Load Tox21
    tox_df = load_tox21_labels()

    # ── Build promiscuity model features (ALL prom data, no tox label needed) ──
    print("\nBuilding promiscuity features...")
    prom_fps, prom_scores, prom_smiles = [], [], []
    for _, row in prom_df.iterrows():
        fp = smiles_to_morgan_fp(row[smiles_col])
        if fp is not None:
            prom_fps.append(fp)
            prom_scores.append(float(row[score_col]))
            prom_smiles.append(row[smiles_col])

    X_prom = np.array(prom_fps)
    y_prom = np.array(prom_scores)
    print(f"  Promiscuity training samples: {len(X_prom)}")

    np.save(os.path.join(output_dir, "X_prom.npy"), X_prom)
    np.save(os.path.join(output_dir, "y_prom.npy"), y_prom)
    pd.DataFrame({"smiles": prom_smiles}).to_csv(
        os.path.join(output_dir, "prom_smiles.csv"), index=False)

    # ── Build toxicity model features (merged data) ──────────────────────────
    print("\nBuilding toxicity features...")

    # Merge on SMILES
    merged = tox_df.merge(prom_df[[smiles_col, score_col]], on='smiles', how='left')
    median_prom = prom_df[score_col].median()
    merged[score_col].fillna(median_prom, inplace=True)

    tox_fps, tox_labels, tox_prom = [], [], []
    for _, row in merged.iterrows():
        fp = smiles_to_morgan_fp(row['smiles'])
        if fp is not None:
            tox_fps.append(fp)
            tox_labels.append(int(row['toxicity']))
            tox_prom.append(float(row[score_col]))

    X_tox_fp = np.array(tox_fps)
    X_tox_prom = np.array(tox_prom).reshape(-1, 1)
    X_tox = np.hstack([X_tox_fp, X_tox_prom])  # 2048 + 1 = 2049 features
    y_tox = np.array(tox_labels)

    print(f"  Toxicity training samples: {len(X_tox)}")
    print(f"  Toxic: {y_tox.sum()} | Non-toxic: {(y_tox==0).sum()}")

    np.save(os.path.join(output_dir, "X_tox.npy"), X_tox)
    np.save(os.path.join(output_dir, "y_tox.npy"), y_tox)

    print(f"\n✅ Features saved to: {output_dir}")
    return X_prom, y_prom, X_tox, y_tox


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_feature_matrix(
        prom_csv=os.path.join(BASE, "data", "promiscuity_with_smiles.csv"),
        output_dir=os.path.join(BASE, "data", "features")
    )
