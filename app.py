import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64
import os
from huggingface_hub import InferenceClient

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SafeByte AI – Food Risk Predictor",
    layout="centered"
)

# ---------------- BACKGROUND IMAGE ----------------
def set_background(image_file):
    # Get absolute path based on the location of this script
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, image_file)

    try:
        with open(full_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}

            h1, h2, h3, p, label {{
                color: white !important;
            }}

            .stTextInput input, .stSelectbox div {{
                background-color: rgba(0,0,0,0.6);
                color: white;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning(f"Background image not found: {full_path}")

# Set your ChatGPT-generated black background
set_background("safebyte.bg.png")  # change the name if your file is different

# ---------------- LOAD MODEL OBJECTS ----------------
model = joblib.load("food_spoilage_model.pkl")
scaler = joblib.load("food_spoilage_scaler.pkl")
encoders = joblib.load("food_spoilage_encoders.pkl")
y_encoder = joblib.load("food_spoilage_y_encoder.pkl")

# ---------------- HUGGING FACE SETUP ----------------
HF_TOKEN = st.secrets["HF_TOKEN"]
client = InferenceClient(token=HF_TOKEN)

# ---------------- DROPDOWN VALUES ----------------
FOOD_ITEMS = encoders["food_item"].classes_.tolist()
VISUAL_SIGNS = encoders["spoilage_visual_signs"].classes_.tolist()
ODOR_SIGNS = encoders["spoilage_odor_signs"].classes_.tolist()
STORAGE_TYPES = encoders["storage_type"].classes_.tolist()
GUIDELINES = encoders["guideline"].classes_.tolist()

# ---------------- UI INPUT ----------------
st.title("🥗 SafeByte AI")
st.caption("Quality Assurance Decision Support System – SDG 3")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    food_item = st.selectbox("Food Item", FOOD_ITEMS)
    visual_sign = st.selectbox("Visual Spoilage Signs", VISUAL_SIGNS)
    odor_sign = st.selectbox("Odor Spoilage Signs", ODOR_SIGNS)
    storage_type = st.selectbox("Storage Type", STORAGE_TYPES)

with col2:
    fridge_days = st.number_input("Observed Fridge Days", 0, 60, 2)
    freezer_days = st.number_input("Observed Freezer Days", 0, 365, 0)
    guideline = st.selectbox("Guideline", GUIDELINES)

# ---------------- PREDICTION ----------------
if st.button("Predict Risk Level"):

    input_df = pd.DataFrame({
        "food_item": [food_item],
        "spoilage_visual_signs": [visual_sign],
        "spoilage_odor_signs": [odor_sign],
        "storage_type": [storage_type],
        "guideline": [guideline],
        "observed_fridge_days": [fridge_days],
        "observed_freezer_days": [freezer_days]
    })

    # Encode categorical features
    for col, le in encoders.items():
        input_df[col] = le.transform(input_df[col])

    # Match training feature order
    input_df = input_df[scaler.feature_names_in_]

    # Scale
    scaled_x = scaler.transform(input_df)

    # Predict
    y_pred_encoded = model.predict(scaled_x)
    y_pred = y_encoder.inverse_transform(y_pred_encoded)[0]

    # ---------------- DISPLAY RISK LEVEL ----------------
    if y_pred.lower() == "high":
        st.markdown(f"<h2 style='color:red'>🧪 Predicted Risk Level: {y_pred}</h2>", unsafe_allow_html=True)
    elif y_pred.lower() == "medium":
        st.markdown(f"<h2 style='color:orange'>🧪 Predicted Risk Level: {y_pred}</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color:green'>🧪 Predicted Risk Level: {y_pred}</h2>", unsafe_allow_html=True)

    # ---------------- AI EXPLANATION ----------------
    messages = [
        {
            "role": "system",
            "content": (
                "You are a food safety expert. Respond with EXACTLY 3 complete sentences in one paragraph. "
                "Do NOT use bullet points, numbering, or headings. Make sure each sentence ends properly with a full stop."
            )
        },
        {
            "role": "user",
            "content": (
                f"Food item: {food_item}\n"
                f"Guideline followed: {guideline}\n"
                f"Predicted risk level: {y_pred}\n\n"
                "Explain the reason for this risk level in simple language."
            )
        }
    ]

    with st.spinner("Generating AI explanation..."):
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=messages,
            max_tokens=140,
            temperature=0.4
        )

    llm_output = response.choices[0].message.content.strip()

    # Safety: ensure sentence closure
    if llm_output and llm_output[-1] not in ".!?":
        llm_output = llm_output.rsplit(" ", 1)[0] + "."

    st.markdown("### 📝 AI Explanation")
    st.write(llm_output)
