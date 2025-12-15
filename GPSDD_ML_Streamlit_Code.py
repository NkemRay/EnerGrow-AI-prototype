import streamlit as st
import pandas as pd
import joblib

st.title("Clean Energy Recommendation for Smallholder Farmers")

# Load trained model
model = joblib.load("clean_energy_recommender_model.joblib")

st.write("Enter basic information to get a clean energy recommendation.")

current_fuel = st.selectbox(
    "Current Cooking / Processing Fuel",
    ["Firewood", "Charcoal", "LPG", "Electricity"]
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
    "Post-Harvest Loss (%)",
    0, 50, 15
)

# Create input in same format as training
input_data = pd.DataFrame([{
    "Current Fuel": current_fuel,
    "Income Level": income,
    "Access to Financing": financing,
    "Post-Harvest Loss (%)": post_harvest_loss
}])

if st.button("Recommend Clean Energy"):
    recommendation = model.predict(input_data)[0]

    st.success(f"🌱 Recommended Clean Energy Technology: **{recommendation}**")

    st.caption(
        "This recommendation is based on patterns learned from similar farmers "
        "in the EnerGrow dataset."
    )

