import streamlit as st
import pandas as pd
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("🕵️ LinkedIn Scam Detector")
input_text = st.text_area("Paste a job post here:")

if st.button("Check"):
    if input_text.strip():
        vector = vectorizer.transform([input_text])
        result = model.predict(vector)[0]
        if result == 1:
            st.error("⚠️ This job might be a SCAM!")
        else:
            st.success("✅ This job looks LEGIT.")
    else:
        st.warning("Please enter some text!")
