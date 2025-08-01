import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
model = joblib.load("best_random_forest_model.pkl")

st.title("Credit Default Risk Prediction")

st.markdown("Enter customer information to predict the likelihood of credit default.")

# Example input features (you can replace these with the actual features from your dataset)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income ($)", min_value=0, value=50000)
debt_ratio = st.slider("Debt Ratio", min_value=0.0, max_value=1.0, value=0.3)
credit_lines = st.number_input("Number of Open Credit Lines", min_value=0, value=5)
late_payments = st.number_input("Number of Late Payments", min_value=0, value=1)

if st.button("Predict Default Risk"):
    input_df = pd.DataFrame({
        'age': [age],
        'income': [income],
        'debt_ratio': [debt_ratio],
        'credit_lines': [credit_lines],
        'late_payments': [late_payments]
    })

    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)[0][1]

    st.write(f"Prediction: {'Likely to Default' if prediction[0] == 1 else 'Not Likely to Default'}")
    st.write(f"Default Probability: {proba:.2%}")