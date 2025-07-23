import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
import nltk
from nltk import tokenize


# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# App version
st.caption("Fake News Detection App — v1.0.1")

# Load the saved models and vectorizer
with open('logistic_regression.pkl', 'rb') as f:
    logistic_regression = pickle.load(f)

with open('decision_tree.pkl', 'rb') as f:
    decision_tree = pickle.load(f)

with open('gradient_boosting.pkl', 'rb') as f:
    gradient_boosting = pickle.load(f)

with open('random_forest.pkl', 'rb') as f:
    random_forest = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorization = pickle.load(f)

# Preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Function to predict news authenticity
def predict_news(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = logistic_regression.predict(new_xv_test)
    pred_DT = decision_tree.predict(new_xv_test)
    pred_GB = gradient_boosting.predict(new_xv_test)
    pred_RF = random_forest.predict(new_xv_test)
    return {
        "Logistic Regression": pred_LR[0],
        "Decision Tree": pred_DT[0],
        "Gradient Boosting": pred_GB[0],
        "Random Forest": pred_RF[0]
    }

# Streamlit app
st.title("📰 Fake News Detection")
st.write("Enter the news text below to check its authenticity:")

news_text = st.text_area("News Text", "")

if st.button("Predict"):
    if news_text.strip():
        predictions = predict_news(news_text)
        st.subheader("Predictions:")
        for model, prediction in predictions.items():
            result = "Real News" if prediction == 1 else "Fake News"
            st.write(f"**{model}**: {result}")
    else:
        st.warning("Please enter some news text before predicting.")
