import joblib
import numpy as np
import streamlit as st

@st.cache_resource
def load_model():
    return joblib.load("credit_card_model.joblib")

model = load_model()

st.title("Credit Card Fraud Detector")
st.write("Live inference demo for financial transaction security.")

# Quick sample auto-filler
if st.button("Load Sample Normal Transaction"):
    st.session_state.amount = 45.0
    st.session_state.time = 1200.0
else:
    if "amount" not in st.session_state:
        st.session_state.amount = 100.0
        st.session_state.time = 0.0

amount = st.number_input("Transaction Amount ($)", value=st.session_state.amount)
time = st.number_input("Transaction Time (Seconds)", value=st.session_state.time)

# Simplified inputs for V1-V28 (Defaults to 0.0)
st.subheader("PCA Features (V1 - V28)")
features = [st.number_input(f"V{i}", value=0.0, format="%.3f") for i in range(1, 29)]

if st.button("Run Fraud Detection"):
    input_data = np.array([time] + features + [amount]).reshape(1, -1)
    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)
    
    if pred[0] == 1:
        st.error(f" Fraudulent Activity Flagged! (Risk Score: {prob[0][1]:.2%})")
    else:
        st.success(f"Legitimate Transaction (Confidence: {prob[0][0]:.2%})")