import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# Load model + encoders + scaler
# ---------------------------------------------------------
MODEL_PATH = "energrow_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"
SCALER_PATH = "scaler.pkl"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, encoders, scaler

try:
    model, label_encoders, scaler = load_artifacts()
except:
    st.error("❌ Model files not found. Upload energrow_model.pkl, label_encoders.pkl, scaler.pkl")
    st.stop()

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🌱 EnerGrow AI – Clean Energy Recommendation System")
st.write("Simple prototype model for recommending clean energy technologies.")

# ---------------------------------------------------------
# Input Form
# ---------------------------------------------------------
with st.form("input_form"):
    st.subheader("Enter Farmer Details")

    col1, col2 = st.columns(2)

    Location = col1.text_input("Location", "Kpaduma")
    AvgTemp = col1.number_input("Avg Temp (°C)", value=30.0)
    CropType = col1.text_input("Crop Type", "tomato")
    HarvestQty = col1.number_input("Harvest Quantity (kg)", value=200.0)

    CurrentFuel = col2.selectbox("Current Fuel", ["firewood", "charcoal", "LPG", "electricity", "none"])
    FuelConsumption = col2.number_input("Fuel Consumption (kg)", value=10.0)
    HealthImpact = col2.number_input("Health Impact Score (0–10)", value=3.0)
    PostHarvestLoss = col2.number_input("Post-Harvest Loss (%)", value=25.0)

    IncomeLevel = col1.number_input("Income Level (₦)", value=50000.0)
    AccessFinancing = col1.selectbox("Access to Financing", ["yes", "no"])
    YouthInvolvement = col1.selectbox("Youth Involvement", ["yes", "no"])

    submitted = st.form_submit_button("Predict Clean Energy Solution")

# ---------------------------------------------------------
# RUN PREDICTION
# ---------------------------------------------------------
if submitted:

    # Build input row
    X = pd.DataFrame([{
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
        "Youth Involvement": YouthInvolvement,
    }])

    # Encode categoricals
    X_encoded = X.copy()
    for col, le in label_encoders.items():
        try:
            X_encoded[col] = le.transform(X_encoded[col])
        except:
            fallback = le.classes_[0]
            st.warning(f"⚠ '{X[col][0]}' not seen during training. Using fallback '{fallback}'")
            X_encoded[col] = le.transform([fallback])[0]

    # Scale numeric columns
    numeric_columns = [c for c in X.columns if c not in label_encoders]
    X_encoded[numeric_columns] = scaler.transform(X_encoded[numeric_columns])

    # Prediction
    pred = model.predict(X_encoded)[0]
    probs = model.predict_proba(X_encoded)[0]
    confidence = round(np.max(probs) * 100, 2)

    st.success(f"### 🌟 Recommended Clean Energy Solution: **{pred}**  
    Confidence: **{confidence}%**")

    # ---------------------------------------------------------
    # Simple Cost–Benefit Model
    # ---------------------------------------------------------
    COSTS = {
        "Solar Dryer": 120000,
        "Efficient Cookstove": 15000,
        "Biogas": 450000,
        "None": 0
    }

    market_price = 50  # ₦ per kg
    annual_savings = (PostHarvestLoss/100) * HarvestQty * 12 * 0.3 * market_price
    annual_savings = round(annual_savings, 2)

    cost = COSTS.get(pred, 0)

    st.write("### 💰 Cost & Savings Estimate")
    st.write(f"Estimated Equipment Cost: **₦{cost:,}**")
    st.write(f"Estimated Annual Savings: **₦{annual_savings:,}**")

    if annual_savings > 0:
        payback_months = round((cost / annual_savings) * 12, 1)
        st.write(f"Estimated Payback Period: **{payback_months} months**")
    else:
        st.write("Payback Period: **Not applicable**")

    # ---------------------------------------------------------
    # Supplier Lookup
    # ---------------------------------------------------------
    SUPPLIERS = [
        {"name": "SolarWorks Nigeria", "tech": "Solar Dryer", "region": "Kpaduma", "price": 120000},
        {"name": "GreenStove Ltd", "tech": "Efficient Cookstove", "region": "Anambra", "price": 20000},
        {"name": "BioGas Co-op", "tech": "Biogas", "region": "Kaduna", "price": 450000},
    ]

    st.write("### 🛒 Suppliers Near You")

    matches = [s for s in SUPPLIERS if s["tech"] == pred]

    if matches:
        for s in matches:
            st.write(f"- **{s['name']}** — ₦{s['price']:,} — Region: {s['region']}")
    else:
        st.info("No supplier found for this technology in the sample database.")

    st.markdown("---")
    st.write("Prototype v1.0 — EnerGrow AI")


