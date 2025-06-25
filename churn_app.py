import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from PIL import Image
import base64

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")


# 🎨 Custom CSS Styling
st.markdown("""
    <style>
        html, body {
            background-color: #e6f0f8;
        }
        .block-container {
            background-color: #e6f0f8 !important;
            padding: 2rem 5rem;
        }
        .center-title {
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            color: #1F4E79;
            margin-top: 0.5rem;
        }
        .center-subtitle {
            text-align: center;
            font-size: 16px;
            color: #3F88C5;
            margin-bottom: 2rem;
        }
        .logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 1rem;
        }
        .logo-container img {
            width: 100%;
            max-width: 400px;
            height: auto;
            object-fit: contain;
        }
        .form-container {
            background-color: #ffffffdd;
            border-radius: 12px;
            padding: 2rem 2.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            margin: 1.5rem auto 2.5rem auto;
        }
        label {
            color: #1F4E79 !important;
            font-weight: 600;
        }
        input, .stNumberInput > div, .stSlider, .stTextInput, .stTextArea, .stSelectbox {
            background-color: #ffffff !important;
            border-radius: 6px;
        }
        div.stButton > button {
            background-color: #1F4E79;
            color: white;
            padding: 0.6rem 2rem;
            font-size: 16px;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            transition: background-color 0.3s ease, transform 0.2s ease-in-out;
        }
        div.stButton > button:hover {
            background-color: #163c5a;
            transform: scale(1.03);
            color: #fff;
        }
        button[aria-label="Increment"], button[aria-label="Decrement"] {
            background-color: #ffffff !important;
            color: #1F4E79 !important;
            border-radius: 6px;
            border: 1px solid #bcd1ea;
            padding: 5px 10px;
            font-weight: bold;
            font-size: 14px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }
        button[aria-label="Increment"]:hover,
        button[aria-label="Decrement"]:hover {
            background-color: #1F4E79 !important;
            color: #ffffff !important;
            border: 1px solid #1F4E79;
            transform: scale(1.05);
        }
        .footer {
            margin-top: 50px;
            text-align: center;
            color: gray;
            font-size: 13px;
        }
    </style>
""", unsafe_allow_html=True)

# 🖼️ Logo Display
logo_path = "Quantumsoft_logo.png"
with open(logo_path, "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()

st.markdown(f"""
    <div class="logo-container">
        <img src="data:image/png;base64,{encoded}" alt="Quantumsoft Logo">
    </div>
""", unsafe_allow_html=True)

# 🧾 Title & Subtitle
st.markdown("<div class='center-title'>Customer Churn Risk Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='center-subtitle'>Enter key behavioral metrics to predict churn risk and get personalized retention strategies.</div>", unsafe_allow_html=True)

# 🔄 Load Trained Model
model = joblib.load("xgb_model_selected_features.joblib")
explainer = shap.Explainer(model)

# 📋 Input Form
st.markdown("<div class='form-container'>", unsafe_allow_html=True)
st.markdown("### 🔢 Enter Customer Metrics:")

app_logins = st.number_input("📱 App Logins", min_value=0)
loans_accessed = st.number_input("💳 Loans Accessed", min_value=0)
loans_taken = st.number_input("💸 Loans Taken", min_value=0)
sentiment_score = st.slider("❤️ Sentiment Score", 0.0, 1.0, 0.5)
web_logins = st.number_input("🌐 Web Logins", min_value=0)
monthly_balance = st.number_input("💰 Monthly Avg Balance", min_value=0.0)

st.markdown("</div>", unsafe_allow_html=True)

# 🔮 Prediction Button
if st.button("🔍 Predict Churn"):
    input_data = np.array([[app_logins, loans_accessed, loans_taken, sentiment_score, web_logins, monthly_balance]])
    churn_prob = model.predict_proba(input_data)[0][1]

    st.metric(label="📊 Predicted Churn Probability", value=f"{churn_prob:.2%}")

    if churn_prob >= 0.6:
        risk = "High Risk"
        action = "📞 Call customer + Offer cashback or upgrade"
    elif churn_prob >= 0.3:
        risk = "Medium Risk"
        action = "📧 Send personalized email + offer"
    else:
        risk = "Low Risk"
        action = "💌 Send thank-you email + loyalty points"

    st.success(f"🛡️ Risk Segment: **{risk}**")
    st.info(f"📌 Recommended Action: **{action}**")

    # ✅ SHAP Explanation (Updated for compatibility)
    st.subheader("🔍 Feature Impact on This Prediction")
    shap_values = explainer(input_data)
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)


# 🦶 Footer
st.markdown("<div class='footer'>© 2025 Quantumsoft Technologies | All rights reserved</div>", unsafe_allow_html=True)
