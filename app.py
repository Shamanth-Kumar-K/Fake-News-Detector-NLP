import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
from nltk import tokenize

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
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Function to predict news authenticity
def predict_news(news):
    testing_news = {"text":[news]}
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
st.title("Fake News Detection")
st.write("Enter the news text below to check its authenticity:")

news_text = st.text_area("News Text", "")

if st.button("Predict"):
    if news_text:
        predictions = predict_news(news_text)
        st.subheader("Predictions:")
        for model, prediction in predictions.items():
            result = "Fake News" if prediction == 0 else "Real News"
            st.write(f"{model}: {result}")
    else:
        st.warning("Please enter some news text.")
