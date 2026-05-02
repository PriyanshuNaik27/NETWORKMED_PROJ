"""
STREAMLIT WEB APP — Drug Toxicity Predictor
Run: streamlit run app.py
NOTE: Run python run_pipeline.py first to train models.
"""

import streamlit as st
import sys, os, json
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

st.set_page_config(page_title="Drug Toxicity Predictor", page_icon="🧪", layout="centered")


# ── Result display (defined first) ───────────────────────────────────────────
def show_result(result: dict):
    score = result["toxicity_score"]
    risk  = result["risk_level"]
    bd    = result.get("score_breakdown", {})

    color_map = {"LOW": "green", "MODERATE": "orange", "HIGH": "red", "VERY HIGH": "darkred"}
    color = next((v for k, v in color_map.items() if k in risk), "gray")

    st.markdown("---")
    st.markdown("### Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Final Toxicity Score", f"{score:.4f}")
    c2.metric("Toxicity %", result["toxicity_percent"])
    c3.metric("Predicted Promiscuity", f"{result.get('predicted_promiscuity', 0):.3f}")

    st.markdown(f"**Risk Level:** :{color}[{risk}]")
    st.info(result["interpretation"])

    # Score breakdown
    if bd:
        with st.expander("📊 Score Breakdown (How was this calculated?)"):
            b1, b2, b3 = st.columns(3)
            b1.metric("ML Model Score", f"{bd.get('ml_model_score', 'N/A')}")

            ld50_score = bd.get("ld50_score")
            ld50_val   = bd.get("ld50_predicted_mg_per_kg")
            if ld50_score is not None:
                b2.metric("LD50 Score", f"{ld50_score:.4f}",
                          help=f"Predicted LD50 = {ld50_val} mg/kg")
            else:
                b2.metric("LD50 Score", "No model")

            alerts = bd.get("structural_alerts", [])
            if alerts:
                b3.metric("Structural Alerts", len(alerts))
                st.error(f"⚠️ Toxic substructures found: **{', '.join(alerts)}**")
            else:
                b3.metric("Structural Alerts", "None ✅")

            if ld50_val:
                if ld50_val <= 5:       cat = "Extremely toxic"
                elif ld50_val <= 50:    cat = "Very toxic"
                elif ld50_val <= 300:   cat = "Toxic"
                elif ld50_val <= 2000:  cat = "Harmful"
                elif ld50_val <= 5000:  cat = "Slightly harmful"
                else:                   cat = "Practically non-toxic"
                st.caption(f"Predicted LD50: **{ld50_val:.1f} mg/kg** — {cat}")

            method = bd.get("scoring_method", "")
            if method:
                st.caption(f"Scoring method: {method}")

    # Molecular properties
    if result.get("molecular_properties"):
        with st.expander("🔬 Molecular Properties"):
            props = result["molecular_properties"]
            pc1, pc2 = st.columns(2)
            for i, (k, v) in enumerate(props.items()):
                (pc1 if i % 2 == 0 else pc2).metric(k.replace("_", " ").title(), v)

    # Lipinski
    lip = result.get("lipinski", {})
    if lip:
        with st.expander("💊 Lipinski Drug-likeness"):
            lc1, lc2 = st.columns(2)
            lc1.metric("Status", lip.get("status", ""))
            lc2.metric("Violations", lip.get("violations_count", 0))
            st.caption(lip.get("interpretation", ""))
            if lip.get("violations"):
                st.warning("Violations: " + ", ".join(lip["violations"]))


# ── Page header ───────────────────────────────────────────────────────────────
st.title("🧪 Drug Toxicity Predictor")
st.markdown("Predict toxicity using **ML model + LD50 model + structural alerts**.")

# ── Check models ──────────────────────────────────────────────────────────────
models_dir = os.path.join(BASE, "models")
if not os.path.exists(os.path.join(models_dir, "toxicity_model.pkl")):
    st.error("❌ Models not found. Run `python run_pipeline.py` first, then refresh.")
    st.stop()

@st.cache_resource
def load_predictor():
    from src.step4_predict import ToxicityPredictor
    return ToxicityPredictor(models_dir)

try:
    predictor = load_predictor()
    ld50_loaded = predictor.ld50_model is not None
    if ld50_loaded:
        st.success("✅ Models loaded | ✅ LD50 prediction model active")
    else:
        st.warning("✅ Models loaded | ⚠️ No LD50 model — place ld50.csv in data/ then run: python run_pipeline.py --train-ld50")
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["🔬 SMILES Input", "💊 Drug Name", "📂 Batch CSV"])

with tab1:
    st.subheader("Predict from SMILES")
    examples = {
        "Aspirin":     "CC(=O)Oc1ccccc1C(=O)O",
        "Naphthalene": "c1ccc2ccccc2c1",
        "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
        "Cyanide":     "[C-]#N",
        "Ethanol":     "CCO",
        "Aniline":     "c1ccc(cc1)N",
    }
    col1, col2 = st.columns([3, 1])
    with col2:
        ex = st.selectbox("Load example", [""] + list(examples.keys()))
    smiles_input = col1.text_input("Enter SMILES:", value=examples.get(ex, ""),
                                   placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")
    if st.button("Predict", key="btn_smiles"):
        if smiles_input.strip():
            with st.spinner("Predicting..."):
                result = predictor.predict(smiles_input.strip())
            if "error" in result:
                st.error(result["error"])
            else:
                show_result(result)
        else:
            st.warning("Please enter a SMILES string.")

with tab2:
    st.subheader("Predict from Drug Name")
    st.caption("Fetches SMILES from PubChem. Requires internet.")
    drug_name_input = st.text_input("Drug name:", placeholder="e.g. Ibuprofen")
    if st.button("Predict by Name", key="btn_name"):
        if drug_name_input.strip():
            with st.spinner(f"Fetching SMILES for '{drug_name_input}'..."):
                result = predictor.predict_from_name(drug_name_input.strip())
            if "error" in result:
                st.error(result["error"])
            else:
                st.info(f"SMILES: `{result['smiles']}`")
                show_result(result)
        else:
            st.warning("Please enter a drug name.")

with tab3:
    st.subheader("Batch Prediction from CSV")
    st.markdown("Upload CSV with `smiles` or `drug_name` column.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        cols_lower = df.columns.str.lower().str.strip().tolist()
        has_smiles = 'smiles' in cols_lower
        has_names  = any(c in cols_lower for c in ['drug_name', 'name'])

        if not has_smiles and not has_names:
            st.error("CSV must have a 'smiles' or 'drug_name' column.")
        else:
            if st.button("Run Batch Prediction"):
                results, prog = [], st.progress(0)
                total = len(df)
                if has_smiles:
                    col = df.columns[cols_lower.index('smiles')]
                    for i, smi in enumerate(df[col]):
                        results.append(predictor.predict(str(smi)))
                        prog.progress((i + 1) / total)
                else:
                    ncol_idx = next(i for i, c in enumerate(cols_lower) if c in ['drug_name', 'name'])
                    ncol = df.columns[ncol_idx]
                    for i, name in enumerate(df[ncol]):
                        results.append(predictor.predict_from_name(str(name)))
                        prog.progress((i + 1) / total)

                out_df = pd.DataFrame([{
                    "input":            r.get("drug_name") or r.get("smiles", ""),
                    "toxicity_score":   r.get("toxicity_score", "error"),
                    "risk_level":       r.get("risk_level", ""),
                    "ml_score":         r.get("score_breakdown", {}).get("ml_model_score", ""),
                    "ld50_score":       r.get("score_breakdown", {}).get("ld50_score", "N/A"),
                    "ld50_mg_per_kg":   r.get("score_breakdown", {}).get("ld50_predicted_mg_per_kg", "N/A"),
                    "ld50_category":    r.get("score_breakdown", {}).get("ld50_category", ""),
                    "structural_alerts": ", ".join(r.get("score_breakdown", {}).get("structural_alerts", [])) or "None",
                    "error":            r.get("error", ""),
                } for r in results])
                st.dataframe(out_df)
                st.download_button("⬇️ Download Results", out_df.to_csv(index=False),
                                   "toxicity_predictions.csv", "text/csv")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## How Scores Work")
    st.markdown("""
    **Three-layer prediction:**

    1. **ML Model** (always runs)
       Tox21-trained classifier using Morgan fingerprints + promiscuity score

    2. **LD50 Model** (if trained)
       Predicts acute toxicity for ANY new SMILES
       Weighted 65% in final score

    3. **Structural Alerts** (always runs)
       Rule-based check for known deadly patterns
       (cyanide, arsenic, mustard gas, etc.)
       Forces score ≥ 0.90 if matched

    **Final score = combination of all available layers**
    """)

    st.markdown("---")
    st.markdown("## Risk Levels")
    st.markdown("""
    🟢 **LOW** — 0.00–0.30

    🟡 **MODERATE** — 0.30–0.55

    🟠 **HIGH** — 0.55–0.75

    🔴 **VERY HIGH** — 0.75–1.00
    """)

    metrics_path = os.path.join(BASE, "models", "training_summary.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            summary = json.load(f)
        # st.markdown("---")
        # st.markdown("## Model Performance")
        # st.metric("Promiscuity R²", summary["promiscuity_model"]["r2"])
        # st.metric("Toxicity ROC-AUC", summary["toxicity_model"]["roc_auc"])

    ld50_metrics_path = os.path.join(BASE, "models", "ld50_model_metrics.json")
    if os.path.exists(ld50_metrics_path):
        with open(ld50_metrics_path) as f:
            ld50_summary = json.load(f)
        # st.metric("LD50 Model R²", ld50_summary["r2"])