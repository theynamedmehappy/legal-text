import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle

# Load pre-trained model and vectorizer
@st.cache_resource
def load_resources():
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("xgb_model.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    return vectorizer, model, label_encoder

vectorizer, model, label_encoder = load_resources()

# App Title and Description
st.title("Legal Text Classification App")
st.write("This app predicts the legal outcome category based on case text.")

# Input Section
st.header("Enter Legal Case Details")
user_input = st.text_area("Case Description", placeholder="Enter case details here...")

offense_options = ["Affirmed", "Cited", "Discussed", "Distinguished", "Followed", "Referred To"]
selected_offense = st.selectbox("Select Common Offense Category", ["None"] + offense_options)

# Prediction Logic
if st.button("Predict Category"):
    if user_input.strip():
        # Vectorize input text
        input_vectorized = vectorizer.transform([user_input])
        
        # Predict using the model
        prediction_encoded = model.predict(input_vectorized)
        prediction = label_encoder.inverse_transform(prediction_encoded)[0]
        
        st.subheader("Prediction Result")
        st.write(f"Predicted Category: **{prediction}**")
        
        # Placeholder for related keywords and penalties
        st.subheader("Related Information")
        st.write("**Keywords:** Placeholder for keywords.")
        st.write("**Related Sections:** Placeholder for legal sections.")
        st.write("**Penalties:** Placeholder for penalties.")
    else:
        st.error("Please enter case details for prediction.")

# Enhancements Section
st.sidebar.header("Enhancements")
st.sidebar.write("- Regional Language Support (Coming Soon)")
st.sidebar.write("- Visual Highlights for Keywords")
st.sidebar.write("- Improved User Interface")