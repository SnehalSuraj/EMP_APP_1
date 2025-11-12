import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="EMI Eligibility & EMI Amount Predictor", layout="wide")
st.title("💰 EMI Eligibility & EMI Amount Predictor")

# Load saved models
@st.cache_resource
def load_models():
    clf_pkg = joblib.load("artifacts/classifier_package.pkl")
    reg_pkg = joblib.load("artifacts/regressor_package.pkl")
    return clf_pkg, reg_pkg

clf_pkg, reg_pkg = load_models()
clf_model = clf_pkg["model_pipeline"]
reg_model = reg_pkg["model_pipeline"]
feature_cols = clf_pkg["feature_cols"]

st.sidebar.header("About this App")
st.sidebar.write("""
This app predicts:
1️⃣ **EMI Eligibility (Classification)**  
2️⃣ **Expected EMI Amount (Regression)**
""")

mode = st.radio("Choose Input Mode", ["Manual Entry", "Upload CSV File"])

def prepare_input(data_dict):
    df = pd.DataFrame([data_dict], columns=feature_cols)
    df = df.fillna(0)
    return df[feature_cols]

if mode == "Manual Entry":
    st.write("Enter feature values below:")
    user_data = {}
    for col in feature_cols:
        user_data[col] = st.number_input(col, value=0.0)
    
    if st.button("Predict"):
        X = prepare_input(user_data)
        pred_class = clf_model.predict(X)[0]
        try:
            proba = clf_model.predict_proba(X)[0]
            st.write(f"🔹 EMI Eligibility: {pred_class} (Probabilities: {proba})")
        except:
            st.write(f"🔹 EMI Eligibility: {pred_class}")
        pred_reg = reg_model.predict(X)[0]
        st.write(f"💸 Predicted EMI Amount: {pred_reg:.2f}")

else:
    file = st.file_uploader("Upload CSV with feature columns", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            preds_class = clf_model.predict(df[feature_cols])
            preds_reg = reg_model.predict(df[feature_cols])
            df["pred_emi_eligibility"] = preds_class
            df["pred_emi_amount"] = preds_reg
            st.dataframe(df.head())
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
