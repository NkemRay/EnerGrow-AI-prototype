# app_full.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import requests
from fpdf import FPDF
import io
import datetime

# -------------------------
# CONFIG
# -------------------------
MODEL_FILE = "energrow_model.pkl"
ENCODERS_FILE = "label_encoders.pkl"
SCALER_FILE = "scaler.pkl"
DATA_FILE = "EnerGrow_Nigeria_MVP_Dataset.csv"
LOG_FILE = "predictions_log.csv"

# Quick conversion factors (example)
CO2_REDUCTION_PER_SAVINGS_USD = 0.0008  # synthetic factor; adjust with real values later

# Example supplier list (replace with your real supplier DB)
SUPPLIERS = [
    {"name": "SolarWorks Nigeria", "tech": "Solar Dryer", "phone": "+2348010000001", "region": "Kpaduma", "price_naira": 120000},
    {"name": "GreenStove Ltd", "tech": "Efficient Cookstove", "phone": "+2348010000002", "region": "Anambra", "price_naira": 20000},
    {"name": "BioGas Co-op", "tech": "Biogas", "phone": "+2348010000003", "region": "Kaduna", "price_naira": 450000},
    {"name": "Local Fitter", "tech": "Efficient Cookstove", "phone": "+2348010000004", "region": "Kpaduma", "price_naira": 15000},
]

# -------------------------
# UTILS: load model + preprocessors
# -------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_FILE)
    encoders = joblib.load(ENCODERS_FILE)
    scaler = joblib.load(SCALER_FILE)
    return model, encoders, scaler

def load_dataset_if_exists():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return None

try:
    model, label_encoders, scaler = load_artifacts()
except Exception as e:
    st.error("Failed to load model artifacts. Ensure energrow_model.pkl, label_encoders.pkl, scaler.pkl are present.")
    st.stop()

df_ref = load_dataset_if_exists()

# -------------------------
# UI: Theme and header
# -------------------------
st.set_page_config(page_title="EnerGrow AI — Full Prototype", layout="wide", page_icon="🌿")

st.markdown(
    """
    <style>
    .reportview-container { background: linear-gradient(180deg, #f7fff7 0%, #ffffff 100%); }
    .stButton>button { background-color: #16a34a; color: white; }
    .large-font { font-size:20px !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("⚡ EnerGrow AI — Clean Energy Recommendation (Full Prototype)")
st.markdown("Prototype for EnerGrow AI: prediction, impact estimates, supplier matching, explainability, and PDF reports.")

# -------------------------
# Sidebar: model insights & class distribution
# -------------------------
st.sidebar.header("Model & Data")
st.sidebar.write("Model: RandomForest prototype")
if df_ref is not None and "Technology Adopted" in df_ref.columns:
    st.sidebar.write("Training class distribution:")
    st.sidebar.bar_chart(df_ref["Technology Adopted"].value_counts())

st.sidebar.markdown("---")
st.sidebar.write("Optional: Provide lat/lon to fetch local weather from Open-Meteo (no API key required).")

# -------------------------
# Input form: farmer data (matching your CSV headers)
# -------------------------
with st.form("farmer_form", clear_on_submit=False):
    st.subheader("Enter farmer / household details")
    c1, c2, c3 = st.columns([1,1,1])

    Location = c1.text_input("Location", value="Kpaduma")
    AvgTemp = c1.number_input("Avg Temp (°C)", value=30.0, step=0.1)
    CropType = c1.text_input("Crop Type", value="tomato")

    HarvestQty = c2.number_input("Harvest Quantity (kg)", value=200.0)
    CurrentFuel = c2.selectbox("Current Fuel", options=["firewood","charcoal","LPG","electricity","none"])
    FuelConsumption = c2.number_input("Fuel Consumption (kg)", value=10.0)

    HealthImpact = c3.number_input("Health Impact Score (0–10)", value=3.0, step=0.1)
    PostHarvestLoss = c3.number_input("Post-Harvest Loss (%)", value=25.0, step=0.1)
    IncomeLevel = c3.number_input("Income Level (₦)", value=50000.0, step=1000.0)

    AccessFinancing = c1.selectbox("Access to Financing", options=["yes","no"])
    YouthInvolvement = c2.selectbox("Youth Involvement", options=["yes","no"])

    # Optional geographic for weather
    lat = c3.text_input("Latitude (optional)", value="")
    lon = c3.text_input("Longitude (optional)", value="")

    submitted = st.form_submit_button("Get Recommendation")

# -------------------------
# Helper: prepare input dataframe and transform
# -------------------------
def prepare_input_df():
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
    return X

# -------------------------
# SHAP helper
# -------------------------
def compute_shap_local(model, X_processed, feature_names):
    try:
        import shap
        explainer = shap.TreeExplainer(model.named_steps['clf'])
        # model expects processed input (numeric array)
        shap_values = explainer.shap_values(X_processed)
        return explainer, shap_values
    except Exception as e:
        return None, None

# -------------------------
# Weather fetch (Open-Meteo) if lat/lon provided
# -------------------------
def fetch_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relativehumidity_2m&daily=temperature_2m_max,temperature_2m_min&timezone=auto"
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
        else:
            return None
    except Exception:
        return None

# -------------------------
# Cost-benefit & CO2 functions
# -------------------------
# Example baseline costs (₦) — replace with real supplier prices
BASE_COSTS = {
    "Solar Dryer": 120000,
    "Efficient Cookstove": 15000,
    "Biogas": 450000,
    "None": 0
}

def estimate_payback(cost_naira, annual_savings_naira):
    if annual_savings_naira <= 0:
        return None
    months = (cost_naira / annual_savings_naira) * 12
    return months

def estimate_co2_reduction(annual_savings_naira):
    # synthetic conversion: savings -> kg CO2 avoided
    # adjust with real conversion factor later
    return round(annual_savings_naira * 0.0002, 2)  # kg CO2/year

# -------------------------
# Supplier matching function
# -------------------------
def match_suppliers(tech, region=None):
    matches = [s for s in SUPPLIERS if s["tech"].lower().replace(" ", "") == tech.lower().replace(" ", "")]
    if region:
        matches = [s for s in matches if s["region"].lower() in region.lower()]
    return matches

# -------------------------
# ACTION on submit: predict, show results, compute extras
# -------------------------
if submitted:
    X_raw = prepare_input_df()
    st.write("### Input summary")
    st.dataframe(X_raw.T)

    # Prepare encoded + scaled input copy
    X_proc = X_raw.copy()

    # 1) Encode categorical fields using label_encoders (they should map exactly those training columns)
    for col, le in label_encoders.items():
        if col in X_proc.columns:
            val = X_proc.loc[0, col]
            try:
                X_proc.loc[0, col] = le.transform([val])[0]
            except Exception:
                # if unseen category, warn and map to a fallback (most frequent class)
                fallback = le.classes_[0]
                st.warning(f"Input category '{val}' for column '{col}' unseen during training. Using fallback '{fallback}'.")
                X_proc.loc[0, col] = le.transform([fallback])[0]

    # 2) Scale numeric features: determine numeric columns by excluding encoder keys
    numeric_cols = [c for c in X_proc.columns if c not in label_encoders]
    try:
        X_proc[numeric_cols] = scaler.transform(X_proc[numeric_cols])
    except Exception as e:
        st.error("Scaler transform failed: " + str(e))
        st.stop()

    # 3) Predict
    try:
        pred = model.predict(X_proc)[0]
        probs = model.predict_proba(X_proc)[0]
        classes = model.classes_
        conf = round(np.max(probs) * 100, 2)
    except Exception as e:
        st.error("Prediction failed: " + str(e))
        st.stop()

    st.success(f"### 🌱 Recommended: **{pred}** (confidence {conf}%)")

    # Show probability table
    prob_df = pd.DataFrame({"Technology": classes, "Probability": probs})
    st.write("#### Prediction probabilities")
    st.dataframe(prob_df.sort_values("Probability", ascending=False).reset_index(drop=True))

    # Cost-benefit estimates
    default_cost = BASE_COSTS.get(pred, 0)
    st.write("### Cost & Payback estimate")
    col_cost, col_savings = st.columns(2)

    # allow user to override cost
    cost_input = col_cost.number_input("Estimated equipment cost (₦)", value=float(default_cost))
    # estimate annual savings as a fraction of cost or from input: we will compute simple formula:
    # Assume annual savings = (PostHarvestLoss% reduction value * HarvestQty * market_price) simplistically.
    # Lacking market price, use a proxy: 50 Naira/kg saved value (adjust to real data)
    market_price_naira_per_kg = 50.0
    saved_kg_per_year = (X_raw.loc[0, "Post-Harvest Loss (%)"] / 100.0) * X_raw.loc[0, "Harvest Quantity (kg)"] * 12 * 0.3
    # 0.3 factor: conservative fraction expected to be saved after tech adoption (example)
    est_annual_savings_naira = round(saved_kg_per_year * market_price_naira_per_kg, 2)
    col_savings.metric("Estimated annual savings (₦)", f"{est_annual_savings_naira}")

    payback_months = estimate_payback(cost_input, est_annual_savings_naira)
    if payback_months is None:
        st.warning("Estimated annual savings are zero or negative; payback cannot be computed.")
    else:
        st.info(f"Estimated payback: ~{payback_months:.1f} months ({payback_months/12:.1f} years)")

    # CO2 estimate
    est_co2 = estimate_co2_reduction(est_annual_savings_naira)
    st.write(f"### Estimated CO₂ reduction: **{est_co2} kg/year** (approximate)")

    # Match suppliers
    st.write("### Suppliers that can provide this technology")
    matches = match_suppliers(pred, X_raw.loc[0, "Location"])
    if matches:
        for s in matches:
            st.write(f"- **{s['name']}** — {s['tech']} — ₦{s['price_naira']:,} — Region: {s['region']} — Phone: {s['phone']}")
    else:
        st.write("No direct matches found in our small supplier directory. Extend the supplier DB for full coverage.")

    # Weather (if lat/lon provided)
    if lat and lon:
        w = fetch_weather(lat, lon)
        if w:
            st.write("### Weather snapshot from Open-Meteo (nearby coordinates)")
            # display simple daily summary
            if 'daily' in w and 'temperature_2m_max' in w['daily']:
                max_temps = w['daily'].get('temperature_2m_max', [])
                min_temps = w['daily'].get('temperature_2m_min', [])
                st.write(f"Next days max temp: {max_temps[:5]}")
                st.write(f"Next days min temp: {min_temps[:5]}")
        else:
            st.write("Weather fetch failed — check latitude/longitude or network.")

    # SHAP explainability (local)
    st.write("### Model explainability (SHAP)")
    try:
        # For shap we need to get processed input used by model pipeline
        # Model is assumed to be a pipeline with preproc and clf
        processed = model.named_steps['preproc'].transform(X_raw.replace(np.nan, 0))
        explainer, shap_vals = None, None
        try:
            import shap
            explainer = shap.TreeExplainer(model.named_steps['clf'])
            shap_vals = explainer.shap_values(processed)
        except Exception as e:
            st.write("SHAP not available for this model or failed: " + str(e))
        if explainer is not None and shap_vals is not None:
            # show top features for the predicted class
            idx = list(model.classes_).index(pred)
            # We will display a simple bar of absolute shap values locally
            shap_arr = np.abs(shap_vals[idx][0])
            # Feature names from preprocessor
            try:
                feat_names = model.named_steps['preproc'].get_feature_names_out()
            except Exception:
                # fallback: numeric + categorical raw names
                feat_names = list(X_raw.columns)
            feat_imp = pd.DataFrame({"feature": feat_names, "importance": shap_arr}).sort_values("importance", ascending=False).head(10)
            st.dataframe(feat_imp)
        else:
            st.write("SHAP explanation not available for this pipeline.")
    except Exception as e:
        st.write("Explainability step failed: " + str(e))

    # -------------------------
    # Generate PDF report (downloadable)
    # -------------------------
    def create_pdf_report(input_df, pred, conf, est_savings, payback_months, est_co2, suppliers_list):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, txt="EnerGrow AI - Recommendation Report", ln=True, align="C")
        pdf.ln(4)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 6, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(4)
        pdf.cell(0, 6, txt="Input summary:", ln=True)
        for k, v in input_df.iloc[0].items():
            pdf.cell(0, 6, txt=f" - {k}: {v}", ln=True)
        pdf.ln(4)
        pdf.cell(0, 6, txt=f"Recommended Technology: {pred} (confidence: {conf}%)", ln=True)
        pdf.cell(0, 6, txt=f"Estimated annual savings (₦): {est_savings}", ln=True)
        if payback_months:
            pdf.cell(0, 6, txt=f"Estimated payback (months): {payback_months:.1f}", ln=True)
        pdf.cell(0, 6, txt=f"Estimated CO2 reduction (kg/year): {est_co2}", ln=True)
        pdf.ln(4)
        pdf.cell(0, 6, txt="Suppliers:", ln=True)
        if suppliers_list:
            for s in suppliers_list:
                pdf.cell(0, 6, txt=f" - {s['name']} | {s['tech']} | ₦{s['price_naira']:,} | {s['phone']}", ln=True)
        else:
            pdf.cell(0, 6, txt=" - No suppliers found in local DB", ln=True)
        # return bytes
        return pdf.output(dest='S').encode('latin-1')

    pdf_bytes = create_pdf_report(X_raw, pred, conf, est_annual_savings_naira, payback_months, est_co2, matches)
    st.download_button("📄 Download recommendation report (PDF)", data=pdf_bytes, file_name="energrow_report.pdf", mime="application/pdf")

    # -------------------------
    # Logging prediction to CSV for later labeling
    # -------------------------
    log_row = X_raw.copy()
    log_row["prediction"] = pred
    log_row["confidence"] = conf
    log_row["estimated_annual_savings_naira"] = est_annual_savings_naira
    log_row["estimated_co2_kg_per_year"] = est_co2
    log_row["timestamp"] = datetime.datetime.now().isoformat()
    if os.path.exists(LOG_FILE):
        log_row.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        log_row.to_csv(LOG_FILE, index=False)
    st.success("Prediction logged to disk for monitoring and future labeling.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.write("EnerGrow AI prototype — built to help farmers choose cost-effective clean energy solutions. Validate all recommendations with local technicians and real-world pilots.")
