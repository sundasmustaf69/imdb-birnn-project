import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("imdb_birnn_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Title
st.title("🎬 IMDb Review Sentiment Classifier")
st.write("Enter a movie review below and click **Predict** to see whether the sentiment is **Positive or Negative**.")

# Input
review = st.text_area("✍️ Your Review", height=150)

# Predict button
if st.button("🚀 Predict"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        # Preprocess
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=100)
        
        # Predict
        pred = model.predict(padded)[0][0]
        sentiment = "✅ Positive" if pred > 0.5 else "❌ Negative"
        confidence = round(float(pred) if pred > 0.5 else 1 - float(pred), 2)

        # Show result
        st.subheader("Prediction Result:")
        st.success(f"**{sentiment}** (Confidence: {confidence})")
