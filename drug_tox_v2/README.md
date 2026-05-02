# 🧪 Drug Toxicity Predictor
### Network Medicine Project

Predict toxicity of any drug compound from its **SMILES string** — no protein target information needed.

---

## 🧠 How It Works

```
New Drug SMILES
      │
      ▼
Morgan Fingerprint (2048 bits)
      │
      ├──► Model 1: Predict Promiscuity Score
      │         (learned from your promiscuity_index data)
      │
      ▼
Fingerprint + Predicted Promiscuity
      │
      ▼
Model 2: Toxicity Score (0–1)
```

**Key insight:** You don't need to know which proteins a new drug binds to. The model learns the relationship between molecular structure and promiscuity from your training data, then uses that predicted promiscuity as a feature for toxicity prediction.

---

## 📁 Project Structure

```
drug_toxicity_project/
│
├── data/
│   ├── promiscuity_index.csv        ← Your original data (drug_name, promiscuity_index)
│   ├── promiscuity_with_smiles.csv  ← After step 1 (adds SMILES column)
│   └── features/                   ← Saved numpy arrays for training
│
├── models/
│   ├── promiscuity_model.pkl        ← Trained promiscuity predictor
│   ├── toxicity_model.pkl           ← Trained toxicity classifier
│   └── training_summary.json        ← Model performance metrics
│
├── src/
│   ├── step1_fetch_smiles.py        ← Drug name → SMILES via PubChem
│   ├── step2_feature_engineering.py ← SMILES → Morgan fingerprints
│   ├── step3_train_models.py        ← Train both ML models
│   └── step4_predict.py             ← Inference pipeline
│
├── run_pipeline.py                  ← 🚀 Run everything from here
├── app.py                           ← 🌐 Streamlit web app
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your data
Place your promiscuity file at:
```
data/promiscuity_index.csv
```
Required columns: `drug_name`, `promiscuity_index`

### 3. Run the full pipeline
```bash
python run_pipeline.py
```
This will:
- Fetch SMILES from PubChem for all your drugs
- Build Morgan fingerprint features
- Train both models
- Show demo predictions

### 4. Launch the web app
```bash
streamlit run app.py
```

---

## 🔬 Predict a New Drug

**From command line:**
```bash
# By SMILES
python run_pipeline.py --smiles "CC(=O)Oc1ccccc1C(=O)O"

# By drug name
python run_pipeline.py --drug "Ibuprofen"
```

**From Python:**
```python
from src.step4_predict import ToxicityPredictor

predictor = ToxicityPredictor()
result = predictor.predict("CC(=O)Oc1ccccc1C(=O)O")
print(result)

# Output:
# {
#   "smiles": "CC(=O)Oc1ccccc1C(=O)O",
#   "toxicity_score": 0.23,
#   "toxicity_percent": "23.0%",
#   "risk_level": "🟢 LOW",
#   "interpretation": "Low toxicity risk...",
#   "predicted_promiscuity": 2.14,
#   "molecular_properties": { ... }
# }
```

---

## 📊 Risk Levels

| Score | Risk Level | Meaning |
|-------|-----------|---------|
| 0.00 – 0.30 | 🟢 LOW | Safe structural profile |
| 0.30 – 0.55 | 🟡 MODERATE | Further validation recommended |
| 0.55 – 0.75 | 🟠 HIGH | Strong structural alerts |
| 0.75 – 1.00 | 🔴 VERY HIGH | High similarity to known toxins |

---

## 🔧 Using Your Own Data from GitHub

Replace `data/promiscuity_index.csv` with your actual file from the repo:
```
https://github.com/PriyanshuNaik27/NETWORKMED_PROJ/tree/main/promiscuity_index
```

The column names are auto-detected — as long as one column has "drug"/"name" and another has "promis"/"index"/"score", it will work.

---

## 📈 Model Details

| Model | Algorithm | Task | Features |
|-------|-----------|------|----------|
| Promiscuity Predictor | Random Forest Regressor | Regression | Morgan FP (2048-bit) |
| Toxicity Classifier | Random Forest / GBM | Binary Classification | Morgan FP + promiscuity score |

Toxicity labels sourced from **Tox21** dataset (~8000 compounds).

---

## 👤 Author
Priyanshu Naik — Network Medicine Project
