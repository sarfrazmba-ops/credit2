import os
import pandas as pd
import joblib
import streamlit as st

DATA_PATH = "credit.csv"
MODEL_PATH = "credit_approval_model.joblib"

st.set_page_config(page_title="Credit Approval Predictor", page_icon="💳", layout="wide")
st.title("Credit Approval Predictor")
st.write("Enter the applicant information to get a credit approval prediction.")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Train it first by running: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
feature_columns = [col for col in df.columns if col != "Approved"]

cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = df.select_dtypes(exclude=["object", "string"]).columns.tolist()

user_inputs = {}

with st.form("prediction_form"):
    for col in feature_columns:
        if col in cat_cols:
            values = sorted([str(v) for v in df[col].dropna().unique()])
            user_inputs[col] = st.selectbox(col, values)
        else:
            median_value = df[col].median()
            user_inputs[col] = st.number_input(
                col,
                value=float(median_value) if pd.notna(median_value) else 0.0,
                step=1.0 if col in ["Age", "Employment Years", "Months At Address", "Num Credit Cards", "Num Late Payments", "Num Inquiries 6M"] else 0.01,
            )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([user_inputs])
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("Approved")
    else:
        st.error("Not Approved")

    st.write("Probability details:")
    for cls, prob in zip(classes, probabilities):
        label = "Approved" if cls == 1 else "Not Approved"
        st.write(f"- {label}: {prob:.2%}")
