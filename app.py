import streamlit as st
import numpy as np
import joblib

# ---------------- LOAD MODEL ----------------
model = joblib.load("laptop_price_model.pkl")
encoders = joblib.load("encoders.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Laptop Price Predictor", page_icon="💻", layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    /* Light gradient background */
    .stApp {
        background: linear-gradient(135deg, #EAF2F8, #D6EAF8, #EBF5FB);
        color: #2C3E50;
    }

    /* Title */
    .title {
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        color: #154360;
    }

    /* Subtitle */
    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #5D6D7E;
        margin-bottom: 30px;
    }

    /* Card */
    .card {
        background-color: #FFFFFF;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.15);
        margin-bottom: 25px;
    }

    /* Section heading */
    .section {
        font-size: 26px;
        font-weight: 700;
        color: #1F618D;
        margin-bottom: 15px;
    }

    /* Result box */
    .result {
        font-size: 30px;
        font-weight: 700;
        color: #145A32;
        background-color: #D4EFDF;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
    }

    /* Footer */
    .footer {
        font-size: 16px;
        text-align: center;
        color: #566573;
        margin-top: 30px;
    }

    /* Input labels */
    label {
        font-size: 18px !important;
        color: #2C3E50 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- HEADER ----------------
st.markdown(
    "<div class='title'>💻 Laptop Price Predictor</div>", unsafe_allow_html=True
)
st.markdown(
    "<div class='subtitle'>Predict laptop prices using Machine Learning</div>",
    unsafe_allow_html=True,
)

# ---------------- INPUT CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown(
    "<div class='section'>🔧 Laptop Specifications</div>", unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Company", encoders["Company"].classes_)
    cpu = st.selectbox("CPU", encoders["Cpu"].classes_)
    ram = st.slider("RAM (GB)", 2, 64, step=2)
    inches = st.slider("Screen Size (Inches)", 10.0, 18.0, step=0.1)

with col2:
    gpu = st.selectbox("GPU", encoders["Gpu"].classes_)
    os = st.selectbox("Operating System", encoders["OpSys"].classes_)
    weight = st.slider("Weight (kg)", 0.5, 5.0, step=0.1)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Price", use_container_width=True):
    comp = encoders["Company"].transform([company])[0]
    cpu_val = encoders["Cpu"].transform([cpu])[0]
    gpu_val = encoders["Gpu"].transform([gpu])[0]
    os_val = encoders["OpSys"].transform([os])[0]

    input_data = np.array([[comp, cpu_val, gpu_val, os_val, ram, weight, inches]])

    price = model.predict(input_data)[0]

    st.markdown(
        f"<div class='result'>💰 Estimated Laptop Price: ₹ {round(price, 2)}</div>",
        unsafe_allow_html=True,
    )

# ---------------- FOOTER ----------------
GITHUB_URL = "https://github.com/Rajputmayuri/Laptop_Price_Prediction_ML_Model"

st.markdown("---")
st.markdown(
    f"<div class='footer'>🔗 <a href='{GITHUB_URL}' target='_blank'>View Project on GitHub</a></div>",
    unsafe_allow_html=True,
)
