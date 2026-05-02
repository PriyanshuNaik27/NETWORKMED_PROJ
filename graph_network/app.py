import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Protein Risk Scanner", page_icon="🧬")
st.title("🧬 Network Medicine: Protein Toxicity Predictor")

# Load your saved data
@st.cache_data
def load_data():
    # index_col=0 is CRUCIAL here to read the names back in
    df = pd.read_csv('protein_features_database.csv', index_col=0)
    # Force names to uppercase for easy searching
    df.index = df.index.astype(str).str.upper().str.strip()
    return df

try:
    data = load_data()

    user_input = st.text_input("Enter Protein Name (e.g., M6PR, FKBP4):", "M6PR")
    protein_input = user_input.upper().strip()

    if st.button('Analyze Risk'):
        if protein_input in data.index:
            prob = data.loc[protein_input, 'risk_score']
            degree = data.loc[protein_input, 'raw_degree']
            
            st.divider()
            col1, col2 = st.columns(2)
            col1.metric(label="Toxicity Risk Probability", value=f"{prob*100:.1f}%")
            col2.metric(label="Network Connections", value=int(degree))
            
            if prob > 0.5:
                st.error(f"🚨 HIGH RISK TARGET: {protein_input}")
                st.write("This protein is a major network hub. Drugs targeting this protein have a high risk of side effects.")
            else:
                st.success(f"✅ LOW RISK TARGET: {protein_input}")
                st.write("This protein is peripherally located. It is likely a more specific and safer drug target.")
        else:
            st.warning(f"Protein '{protein_input}' not found in the database.")
            st.write("**Try these examples:** " + ", ".join(list(data.index[:5])))

except Exception as e:
    st.error(f"Error loading files: {e}")
    