import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# --- Load artifacts ---
@st.cache_resource
def load_artifacts():
    model = joblib.load("Titanic_pred_model.pkl")
    scaler = joblib.load("titanic_scaler.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, scaler, model_columns

model, scaler, model_columns = load_artifacts()

st.title("🚢 Titanic Survival Predictor")
st.write("Predict survival chances based on passenger info.")

# --- Input form ---
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.number_input("Age", 0.0, 100.0, 30.0)
        sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
    with col2:
        parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
        fare = st.number_input("Fare", 0.0, 1000.0, 32.204)
        embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
        submit = st.form_submit_button("Predict Survival")

def preprocess_input(single_input: dict, model_columns):
    df = pd.DataFrame([single_input])
    df_encoded = pd.get_dummies(df, drop_first=True)
    df_reindexed = df_encoded.reindex(columns=model_columns, fill_value=0)
    return df_reindexed

if submit:
    input_dict = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }
    X_input = preprocess_input(input_dict, model_columns)
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1] if hasattr(model, "predict_proba") else None

    if pred == 1:
        st.success(f"✅ Survived (Probability: {proba:.2f})" if proba else "✅ Survived")
    else:
        st.error(f"❌ Did NOT Survive (Probability: {proba:.2f})" if proba else "❌ Did NOT Survive")

    st.info("Prediction made successfully!")
