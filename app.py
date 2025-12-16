import streamlit as st
import numpy as np
import joblib

# Load model & encoders
model = joblib.load("laptop_price_model.pkl")
encoders = joblib.load("encoders.pkl")

# Page config
st.set_page_config(
    page_title="Laptop Price Predictor", page_icon="💻", layout="centered"
)

# Title
st.markdown(
    "<h1 style='text-align: center;'>💻 Laptop Price Predictor</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align: center;'>Predict laptop prices using Machine Learning</p>",
    unsafe_allow_html=True,
)

st.divider()

# ---- INPUT SECTION ----
st.subheader("🔧 Laptop Specifications")

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

st.divider()

# ---- PREDICTION ----
if st.button("🔮 Predict Price", use_container_width=True):
    comp = encoders["Company"].transform([company])[0]
    cpu_val = encoders["Cpu"].transform([cpu])[0]
    gpu_val = encoders["Gpu"].transform([gpu])[0]
    os_val = encoders["OpSys"].transform([os])[0]

    input_data = np.array([[comp, cpu_val, gpu_val, os_val, ram, weight, inches]])

    price = model.predict(input_data)[0]

    st.success(f"💰 Estimated Laptop Price: ₹ {round(price, 2)}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>"
    "<a href='https://github.com/your-username/Laptop-Price-Predictor' target='_blank'>"
    "🐙 GitHub Repository</a></p>",
    unsafe_allow_html=True,
)
