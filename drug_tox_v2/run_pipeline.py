"""
MAIN PIPELINE RUNNER
Runs everything:
  Step 1: Fetch SMILES from PubChem
  Step 2: Build feature matrix
  Step 3: Train Tox21 ML models
  Step 4: Train LD50 model (if data/ld50.csv present)
  Step 5: Demo predictions

Usage:
  python run_pipeline.py                  # full pipeline
  python run_pipeline.py --predict-only   # skip training
  python run_pipeline.py --smiles "CCO"   # predict single SMILES
  python run_pipeline.py --drug "Aspirin" # predict by name
  python run_pipeline.py --train-ld50     # retrain LD50 model only
"""

import argparse, os, sys, json
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

DATA_DIR     = os.path.join(BASE, "data")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
MODEL_DIR    = os.path.join(BASE, "models")
PROM_CSV     = os.path.join(DATA_DIR, "promiscuity_with_smiles.csv")
RAW_CSV      = os.path.join(DATA_DIR, "promiscuity_index.csv")
LD50_CSV     = os.path.join(DATA_DIR, "ld50.csv")


def step1():
    if os.path.exists(PROM_CSV):
        print("✅ SMILES already fetched."); return
    print("\n📡 STEP 1: Fetching SMILES from PubChem...")
    from src.step1_fetch_smiles import fetch_smiles_for_dataset
    fetch_smiles_for_dataset(RAW_CSV, PROM_CSV)

def step2():
    if os.path.exists(os.path.join(FEATURES_DIR, "X_tox.npy")):
        print("✅ Features already built."); return
    print("\n🔬 STEP 2: Building feature matrix...")
    from src.step2_feature_engineering import build_feature_matrix
    build_feature_matrix(PROM_CSV, FEATURES_DIR)

def step3():
    if os.path.exists(os.path.join(MODEL_DIR, "toxicity_model.pkl")):
        print("✅ Tox21 models already trained."); return
    print("\n🤖 STEP 3: Training Tox21 models...")
    from src.step3_train_models import run_training
    run_training(FEATURES_DIR, MODEL_DIR)

def step4_ld50(force=False):
    ld50_model_path = os.path.join(MODEL_DIR, "ld50_model.pkl")
    if os.path.exists(ld50_model_path) and not force:
        print("✅ LD50 model already trained."); return
    if not os.path.exists(LD50_CSV):
        print("ℹ️  No ld50.csv found — skipping LD50 model training.")
        print("   Place ld50.csv in data/ and re-run for better predictions.")
        return
    print("\n💊 STEP 4: Training LD50 prediction model...")
    from src.step5_train_ld50_model import train_ld50_model
    metrics = train_ld50_model(LD50_CSV, MODEL_DIR)
    print(f"   LD50 model R² (log scale): {metrics['r2_log_scale']}")

def demo():
    print("\n🧪 Demo predictions...")
    from src.step4_predict import ToxicityPredictor
    pred = ToxicityPredictor(MODEL_DIR)

    tests = [
        ("Aspirin",     "CC(=O)Oc1ccccc1C(=O)O"),
        ("Cyanide",     "[C-]#N"),
        ("Naphthalene", "c1ccc2ccccc2c1"),
        ("Paracetamol", "CC(=O)Nc1ccc(O)cc1"),
        ("Ethanol",     "CCO"),
    ]
    print(f"\n{'Drug':<14} {'Score':>7} {'Risk':<20} {'Method'}")
    print("-"*65)
    for name, smi in tests:
        r  = pred.predict(smi, drug_name=name)
        bd = r["score_breakdown"]
        print(f"{name:<14} {r['toxicity_score']:>7.4f} {r['risk_level']:<20} {bd['scoring_method']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-only",  action="store_true")
    parser.add_argument("--train-ld50",    action="store_true")
    parser.add_argument("--smiles",        type=str, default=None)
    parser.add_argument("--drug",          type=str, default=None)
    parser.add_argument("--retrain",       action="store_true")
    args = parser.parse_args()

    # Single prediction mode
    if args.smiles or args.drug:
        from src.step4_predict import ToxicityPredictor
        pred   = ToxicityPredictor(MODEL_DIR)
        result = pred.predict(args.smiles) if args.smiles else pred.predict_from_name(args.drug)
        print(json.dumps(result, indent=2))
        return

    # Retrain LD50 model only
    if args.train_ld50:
        step4_ld50(force=True)
        return

    # Full pipeline
    if not args.predict_only:
        step1()
        step2()
        step3()
        step4_ld50()

    demo()
    print("\n🎉 Done! Run `streamlit run app.py` to launch the web app.")
    print('💡 Predict new drug: python run_pipeline.py --smiles "YOUR_SMILES"')

if __name__ == "__main__":
    main()
