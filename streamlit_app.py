
import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

# Load model once
model = joblib.load("tox21_model_v1.pkl")


def featurize(smiles: str):
    """Convert SMILES into fingerprints + descriptors for model input."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Morgan fingerprint (ECFP4, 2048 bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_array = np.array(fp)

    extra_features = np.array([
        Descriptors.MolWt(mol),       # Molecular Weight
        Descriptors.MolLogP(mol),     # LogP
        Descriptors.NumHAcceptors(mol), 
        Descriptors.NumHDonors(mol),
        Descriptors.TPSA(mol)         # Topological Polar Surface Area
    ])

    # # Compact descriptor set (same as training)
    # calc = MoleculeDescriptors.MolecularDescriptorCalculator(
    #     [d[0] for d in Descriptors._descList]
    # )
    # desc_array = np.array(calc.CalcDescriptors(mol))

    # Combine into a single feature vector
    # return np.concatenate([fp_array, desc_array])

    return np.concatenate([fp_array, extra_features])



st.title("Notebook 1: Tox21 Predictor")
st.write("Enter a SMILES string to get a prediction from the trained model.")

# Input box
smiles = st.text_input("Input SMILES")

if st.button("Predict") and smiles:

    features = featurize(smiles)
    if features is not None:
        try:
            prediction = model.predict([features])
            if prediction[0] == 1:
                st.success("The compound is predicted to be toxic.")
            elif prediction[0] == 0:
                st.success("The compound is predicted to be non-toxic.")
            #st.success(f"Prediction: {prediction[0]}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.error("Invalid SMILES string")
