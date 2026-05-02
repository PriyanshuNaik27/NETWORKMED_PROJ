"""
STEP 1: Drug Name → SMILES via PubChem API
==========================================
Input:  promiscuity_index.csv  (drug_name, promiscuity_index)
Output: promiscuity_with_smiles.csv (drug_name, promiscuity_index, smiles)
"""

import requests
import pandas as pd
import time
import os

def get_smiles_from_pubchem(drug_name: str) -> str | None:
    """Fetch canonical SMILES for a drug name from PubChem."""
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{requests.utils.quote(drug_name)}/property/IsomericSMILES/JSON"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            props = data['PropertyTable']['Properties'][0]
            return props.get('IsomericSMILES') or props.get('SMILES')
    except Exception:
        pass
    return None


def fetch_smiles_for_dataset(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Detect drug name column (flexible)
    name_col = next((c for c in df.columns if 'drug' in c or 'name' in c or 'compound' in c), df.columns[0])
    print(f"Using '{name_col}' as drug name column")
    print(f"Total drugs: {len(df)}")

    smiles_list = []
    found = 0

    for i, name in enumerate(df[name_col]):
        smiles = get_smiles_from_pubchem(str(name))
        smiles_list.append(smiles)
        if smiles:
            found += 1
        time.sleep(0.2)  # PubChem rate limit: be polite

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(df)} | Found: {found}")

    df['smiles'] = smiles_list
    df_clean = df.dropna(subset=['smiles']).reset_index(drop=True)

    df_clean.to_csv(output_csv, index=False)
    print(f"\n✅ Done! {found}/{len(df)} drugs matched ({found/len(df)*100:.1f}%)")
    print(f"Saved to: {output_csv}")
    return df_clean


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fetch_smiles_for_dataset(
        input_csv=os.path.join(BASE, "data", "promiscuity_index.csv"),
        output_csv=os.path.join(BASE, "data", "promiscuity_with_smiles.csv")
    )
