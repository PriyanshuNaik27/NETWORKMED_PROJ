# 🧪 Drug Toxicity Predictor
### Network Medicine Project Submission

---

## Team Members

| Name | Roll Number |
|------|-------------|
| Manish Hingar | 230106040 |
| Aditya Dhaniyaal | 230106002 |
| Priyanshu Naik | 230106052 |

---

## What This Project Does

This project predicts how toxic a drug compound is — using only its **SMILES string** as input. No protein target information is needed.

A SMILES string is a text representation of a molecule's structure. For example:
- Aspirin → `CC(=O)Oc1ccccc1C(=O)O`
- Cyanide → `[C-]#N`
- Paracetamol → `CC(=O)Nc1ccc(O)cc1`

You give the model a SMILES string, and it gives back a **toxicity score from 0 to 1** along with a risk level: LOW, MODERATE, HIGH, or VERY HIGH.

---

## The Core Idea

Most toxicity tools require you to know which proteins a drug binds to. We don't have that information for a new drug. So instead, we built a pipeline that:

1. Converts the molecule's structure into numbers (Morgan fingerprints)
2. Uses those numbers to predict how many proteins the drug likely binds to (promiscuity)
3. Uses fingerprints + promiscuity to predict toxicity
4. Also predicts LD50 (lethal dose) from structure for acute toxicity
5. Checks for known deadly chemical patterns (cyanide, arsenic, mustard gas, etc.)
6. Combines all three into one final score

---

## Three-Layer Scoring System

### Layer 1 — Tox21 ML Model (35% weight)
Trained on the Tox21 dataset from NIH (8,000 compounds, 12 toxicity tests). Predicts whether a compound triggers endocrine disruption, DNA damage, oxidative stress, or cancer pathways.

### Layer 2 — LD50 Prediction Model (65% weight)
Trained on 7,396 compounds with known LD50 values. LD50 is the dose that kills 50% of test subjects — it directly measures acute toxicity. This model predicts LD50 for ANY new SMILES, not just compounds in the database.

### Layer 3 — Structural Alerts (Override)
A rule-based check for 14 known deadly chemical substructures. If any are found, the score is forced to 0.90 or higher regardless of what the ML models say. This catches compounds like cyanide that score low on Tox21 because Tox21 does not test for acute poisoning.

### Final Score Formula
```
If structural alert detected:
    final_score = max(0.90, ml_score)

Else if LD50 model available:
    final_score = 0.35 × ml_score + 0.65 × ld50_score

Else:
    final_score = ml_score
```

### Risk Levels
```
0.00 – 0.30  →  🟢 LOW
0.30 – 0.55  →  🟡 MODERATE
0.55 – 0.75  →  🟠 HIGH
0.75 – 1.00  →  🔴 VERY HIGH
```

---

## Datasets

### 1. Promiscuity Index (Custom)
- 2,339 drug compounds with their promiscuity index (number of protein targets)
- Built from network medicine analysis
- Columns: drug_name, promiscuity_index
- SMILES are fetched automatically from PubChem API

### 2. Tox21
- ~8,000 compounds with 12 binary toxicity labels
- Source: NIH Tox21 Challenge
- Download: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz
- Place at: data/tox21.csv

The 12 toxicity tasks are:

| Task | What It Tests |
|------|---------------|
| NR-AR | Androgen receptor (hormone disruption) |
| NR-AhR | Chemical stress response |
| NR-Aromatase | Estrogen production |
| NR-ER | Estrogen receptor |
| NR-PPAR-gamma | Fat metabolism |
| SR-ARE | Oxidative stress |
| SR-ATAD5 | DNA damage |
| SR-HSE | Heat shock / protein stress |
| SR-MMP | Mitochondrial damage |
| SR-p53 | Tumour suppressor / cancer |

All 12 are aggregated into one label: toxic if ANY task is 1, non-toxic if all are 0.

### 3. LD50 Dataset
- 7,396 compounds
- LD50 values in log10(mg/kg) scale
- Example: value 2.265 means 10^2.265 = 184 mg/kg
- Place at: data/ld50.csv

LD50 to toxicity score conversion:

| LD50 (mg/kg) | Category |
|---|---|
| ≤ 5 | Extremely toxic |
| ≤ 50 | Very toxic |
| ≤ 300 | Toxic |
| ≤ 2000 | Harmful |
| ≤ 5000 | Slightly harmful |
| > 5000 | Practically non-toxic |

---

## Algorithms Used

### Morgan Fingerprints
The core featurization method. Converts a SMILES string into a 2048-bit binary vector where each bit represents the presence or absence of a specific circular substructure around each atom. Implemented with RDKit using radius=2.

```python
from rdkit.Chem import AllChem
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
```

### Random Forest
Used for both the promiscuity predictor and toxicity classifier. An ensemble of decision trees that votes on the final prediction. Handles high-dimensional sparse data (like fingerprints) well without overfitting.

### Gradient Boosting
Also tried for the toxicity classifier. Builds trees sequentially where each tree corrects errors of the previous one. The model with better ROC-AUC on the test set is automatically selected.

### SMARTS Pattern Matching
Used for structural alerts. SMARTS is a chemical pattern language. RDKit checks if the molecule contains any of 14 known dangerous substructures.

```
Cyanide:     [C-]#N or C#N
Arsenic:     [As]
Mercury:     [Hg]
Mustard gas: ClCCSCCCl
Phosgene:    ClC(=O)Cl
Nitro group: [N+](=O)[O-]
Epoxide:     C1OC1
... and more
```

---

## Project Structure

```
drug_toxicity_project/
│
├── run_pipeline.py              ← Run this first
├── app.py                       ← Streamlit web app
├── requirements.txt
│
├── data/
│   ├── promiscuity_index.csv    ← Already included
│   ├── tox21.csv                ← Download separately
│   ├── ld50.csv                 ← Your LD50 dataset
│   ├── promiscuity_with_smiles.csv  ← Auto-generated
│   └── features/                ← Auto-generated
│
├── models/
│   ├── promiscuity_model.pkl    ← Trained automatically
│   ├── toxicity_model.pkl       ← Trained automatically
│   ├── ld50_model.pkl           ← Trained with --train-ld50
│   └── training_summary.json
│
└── src/
    ├── step1_fetch_smiles.py         ← Drug name → SMILES
    ├── step2_feature_engineering.py  ← SMILES → features
    ├── step3_train_models.py         ← Train Tox21 models
    ├── step4_predict.py              ← Prediction pipeline
    └── step5_train_ld50_model.py     ← Train LD50 model
```

---

## Installation

```bash
git clone https://github.com/PriyanshuNaik27/NETWORKMED_PROJ
cd drug_tox_v2
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### Install packages in this exact order
```bash
pip install --upgrade pip
pip install rdkit
pip install scikit-learn numpy pandas joblib requests
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install streamlit
```

### Verify installation
```bash
python -c "from rdkit import Chem; print('RDKit OK')"
python -c "import sklearn; print('Sklearn OK')"

```

---




###   Run the full pipeline
```bash
python run_pipeline.py
```
This fetches SMILES, builds features, and trains both models automatically.




### Step 5 — Launch the web app
```bash
streamlit run app.py
```
Open browser at: http://localhost:8501

---

## Predict from Command Line

```bash
# By SMILES string
python run_pipeline.py --smiles "CC(=O)Oc1ccccc1C(=O)O"

# By drug name (fetches SMILES from PubChem)
python run_pipeline.py --drug "Ibuprofen"
```

---

## Example Predictions

| Drug | Score | Risk |
|------|-------|------|
| Aspirin | ~0.18 | 🟢 LOW |
| Paracetamol | ~0.22 | 🟢 LOW |
| Ethanol | ~0.15 | 🟢 LOW |
| Naphthalene | ~0.65 | 🟠 HIGH |
| Aniline | ~0.58 | 🟠 HIGH |
| Cyanide | ~0.92 | 🔴 VERY HIGH |
| Mustard gas | ~0.91 | 🔴 VERY HIGH |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named 'rdkit'` | Run `conda activate toxicity` first |
| `Found: 0` during SMILES fetch | Internet issue or PubChem rate limit — try again |
| `ld50_lookup` attribute error | Update app.py line: `predictor.ld50_model is not None` |
| Models not found | Run `python run_pipeline.py` first |
| DeepChem install fails | You are on Python 3.11+ — use conda with Python 3.10 |
| Tox21 not loading | Place tox21.csv in data/ folder |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10 | Language |
| rdkit | >=2023.3.1 | Chemistry toolkit — fingerprints, descriptors |
| scikit-learn | >=1.3.0 | ML models |
| numpy | >=1.24.0 | Array operations |
| pandas | >=2.0.0 | Data handling |
| joblib | >=1.3.0 | Model saving |
| torch | >=2.0.0 | Required by DeepChem |
| streamlit | >=1.28.0 | Web app |
| requests | >=2.31.0 | PubChem API |
