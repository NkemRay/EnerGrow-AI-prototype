import streamlit as st
import pandas as pd
import joblib

st.title("Clean Energy Recommendation for Smallholder Farmers")

# Load trained model
model = joblib.load("clean_energy_recommender_model.joblib")

st.write("Enter basic information to get a clean energy recommendation.")

# -----------------------------
# User Inputs (NO fuel type)
# -----------------------------
fuel_consumption = st.number_input(
    "Average Fuel Consumption (kg per month)",
    min_value=0.0,
    value=30.0,
    help="Typical monthly fuel used for cooking or processing"
)

income = st.selectbox(
    "Income Level",
    ["Low", "Medium", "High"]
)

financing = st.selectbox(
    "Access to Financing",
    ["Yes", "No"]
)

post_harvest_loss = st.slider(
    "Typical Post-Harvest Loss (%)",
    0, 50, 15
)

# -----------------------------
# INTERNAL ASSUMPTION
# -----------------------------
# We assume firewood as baseline fuel for prototype
ASSUMED_FUEL = "Firewood"

input_data = pd.DataFrame([{
    "Current Fuel": ASSUMED_FUEL,
    "Income Level": income,
    "Access to Financing": financing,
    "Post-Harvest Loss (%)": post_harvest_loss
}])

# -----------------------------
# Impact assumptions
# -----------------------------
LOSS_REDUCTION_RATES = {
    "Solar Dryer": 0.40,
    "Solar Cold Storage": 0.60,
    "Improved Cookstove": 0.10,
    "Efficient LPG Stove": 0.05,
    "Solar Home System": 0.15
}

EMISSION_FACTOR_FIREWOOD = 0.0018  # tCO₂ per kg
CLEAN_TECH_EMISSION = {
    "Improved Cookstove": 0.9,
    "Efficient LPG Stove": 1.0,
    "Solar Dryer": 0.2,
    "Solar Cold Storage": 0.3,
    "Solar Home System": 0.1
}

AVERAGE_PRODUCE_VALUE = 500_000  # NGN per season

# -----------------------------
# Prediction + Impact
# -----------------------------
if st.button("Recommend Clean Energy"):
    recommendation = model.predict(input_data)[0]

    st.success(f"🌱 Recommended Clean Energy Technology: **{recommendation}**")

    # Loss reduction
    reduction_rate = LOSS_REDUCTION_RATES.get(recommendation, 0.1)
    loss_reduced = post_harvest_loss * reduction_rate
    income_gain = (loss_reduced / 100) * AVERAGE_PRODUCE_VALUE

    # CO₂ savings using consumption
    current_emissions = fuel_consumption * EMISSION_FACTOR_FIREWOOD
    clean_emissions = CLEAN_TECH_EMISSION.get(recommendation, 0.2)
    co2_saved = max(current_emissions - clean_emissions, 0)

    # Display
    st.subheader("🌍 Estimated Impact")

    col1, col2, col3 = st.columns(3)
    col1.metric("📉 Loss Reduction", f"{loss_reduced:.1f}%")
    col2.metric("💰 Income Increase", f"₦{income_gain:,.0f}")
    col3.metric("🌱 CO₂ Savings", f"{co2_saved:.2f} tCO₂/year")

    st.caption(
        "Fuel type is assumed to be firewood for this prototype. "
        "Impact values are indicative estimates."
    )



