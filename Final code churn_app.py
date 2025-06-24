import streamlit as st
import numpy as np
import joblib
from PIL import Image
import base64

# 🌐 Page Configuration
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")


# 🎨 Custom CSS Styling
st.markdown("""
    <style>
        /* Full screen background */
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
            margin: 0 0 0.5rem 0;
        }

        .logo-container img {
            width: 2000px;
            height: 150px;
            object-fit: contain;
        }

        .footer {
            margin-top: 50px;
            text-align: center;
            color: gray;
            font-size: 13px;
        }

        label {
            color: #1F4E79 !important;
            font-weight: 600;
        }

        /* ❌ REMOVE this to fix red box issue */
        /* 
        .stSlider > div[data-baseweb="slider"] > div {
            background: #E04E5A;
        } 
        */
    </style>
""", unsafe_allow_html=True)



# 🖼️ Centered Logo (with height and width)
logo_path = "Quantumsoft_logo.png"
with open(logo_path, "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()

st.markdown(f"""
    <div class="logo-container">
        <img src="data:image/png;base64,{encoded}" alt="Quantumsoft Logo">
    </div>
""", unsafe_allow_html=True)

# 🧾 Title and Subtitle
st.markdown("<div class='center-title'>Customer Churn Risk Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='center-subtitle'>Enter key behavioral metrics to predict churn risk and get personalized retention strategies.</div>", unsafe_allow_html=True)

# 🔄 Load Trained Model
model = joblib.load("xgb_model_selected_features.joblib")

# 📋 Input Form
st.markdown("### 🔢 Enter Customer Metrics:")
app_logins = st.number_input("📱 App Logins", min_value=0)
loans_accessed = st.number_input("💳 Loans Accessed", min_value=0)
loans_taken = st.number_input("💸 Loans Taken", min_value=0)
sentiment_score = st.slider("❤️ Sentiment Score", 0.0, 1.0, 0.5)
web_logins = st.number_input("🌐 Web Logins", min_value=0)
monthly_balance = st.number_input("💰 Monthly Avg Balance", min_value=0.0)

# 🔮 Prediction
if st.button("🔍 Predict Churn"):
    input_data = np.array([[app_logins, loans_accessed, loans_taken, sentiment_score, web_logins, monthly_balance]])
    churn_prob = model.predict_proba(input_data)[0][1]

    st.metric(label="📊 Predicted Churn Probability", value=f"{churn_prob:.2%}")

    # 🎯 Logic
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

# 🦶 Footer
st.markdown("<div class='footer'>© 2025 Quantumsoft Technologies | All rights reserved</div>", unsafe_allow_html=True)
