import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("models/churn_model.pkl")
features = joblib.load("models/model_features.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title(" Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn")

st.subheader("Customer Details")

user_input = {}
for feature in features:
    user_input[feature] = st.number_input(feature, value=0.0)

input_df = pd.DataFrame([user_input])

if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"Customer is likely to churn")
    else:
        st.success(f"Customer is not likely to churn")

    st.write(f"**Churn Probability:** {probability:.2f}")

    # Feature importance visualization
    st.subheader("Top Factors Affecting Churn")

    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(10)

    st.bar_chart(importance)