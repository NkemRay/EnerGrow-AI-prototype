import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ======================
# LOAD ARTIFACTS
# ======================
@st.cache_resource
def load_all():
    model = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, encoders, scaler

model, encoders, scaler = load_all()

st.title("⚡ EnerGrow AI – Clean Energy Recommendation")
st.write("Provide the household/farm details below:")

# ======================
# USER INPUT FORM
# ======================
with st.form("input_form"):
    Location = st.text_input("Location")
    AvgTemp = st.number_input("Avg Temperature (°C)")
    CropType = st.text_input("Crop Type")
    HarvestQty = st.number_input("Harvest Quantity (kg)")
    CurrentFuel = st.text_input("Current Fuel Type")
    FuelConsumption = st.number_input("Fuel Consumption (kg)")
    HealthImpact = st.number_input("Health Impact Score (0–10)")
    PostHarvestLoss = st.number_input("Post-Harvest Loss (%)")
    IncomeLevel = st.number_input("Income Level (₦)")
    AccessFinancing = st.text_input("Access to Financing (yes/no)")
    YouthInvolvement = st.text_input("Youth Involvement (yes/no)")

    submit = st.form_submit_button("Predict")

# ======================
# MAKE PREDICTION
# ======================
if submit:
    input_dict = {
        "Location": Location,
        "Avg Temp (Â°C)": AvgTemp,
        "Crop Type": CropType,
        "Harvest Quantity (kg)": HarvestQty,
        "Current Fuel": CurrentFuel,
        "Fuel Consumption (kg)": FuelConsumption,
        "Health Impact Score": HealthImpact,
        "Post-Harvest Loss (%)": PostHarvestLoss,
        "Income Level": IncomeLevel,
        "Access to Financing": AccessFinancing,
        "Youth Involvement": YouthInvolvement
    }

    X = pd.DataFrame([input_dict])

    # Encode categorical
    for col, encoder in encoders.items():
        X[col] = encoder.transform([X[col].iloc[0]])

    # Scale numeric
    numeric_cols = [c for c in X.columns if c not in encoders]
    X[numeric_cols] = scaler.transform(X[numeric_cols])

    # Predict
    pred = model.predict(X)[0]
    st.success(f"🌿 Recommended Clean Energy Technology: **{pred}**")

