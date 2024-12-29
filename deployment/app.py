import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer


#repurposed from the rs notebooks :)
# remove special characters from text
def clean_text(text):
    if isinstance(text, str):
        return re.sub(r'[^\w\s]', '', text).strip()
    return text

# load and preprocess the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Description'] = df['Description'].str.lower().apply(clean_text)
    return df

# prepare tf-idf vectorizer and matrix
def prepare_tfidf(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Description'])
    return vectorizer, tfidf_matrix

# search descriptions based on query
def search_descriptions(query, df, vectorizer, tfidf_matrix, top_n=5):
    query_vec = vectorizer.transform([query.lower()])
    similarity_scores = np.dot(query_vec, tfidf_matrix.T).toarray()[0]
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]
    results = df.iloc[top_indices].copy()
    results['similarity_score'] = similarity_scores[top_indices] * 100  # convert to percentage
    return results

st.image("deployment/invest.eye.png")

st.title("Invest.eye: your eye into a smart investment")
st.write("welcome to invest.eye! this tool helps you explore and evaluate public investment fund (pif) companies and their plans. use it to gain insights and make informed investment decisions.")

file_path = 'deployment/pif_companies_filtered.csv'

df = load_data(file_path)

# prepare tf-idf data
vectorizer, tfidf_matrix = prepare_tfidf(df)

# search bar for queries
query = st.text_input("search for the company you want to explore")

if query:
    # get results ( retrieving)
    results = search_descriptions(query, df, vectorizer, tfidf_matrix)

    st.write(f"### top results for: {query}")
    for _, row in results.iterrows():
        st.write(f"{row['Title']}, {row['Description']}")
        st.write(f"**match percentage:** {row['similarity_score']:.2f}% | **source:** PIF - {row['Title']}")
        st.write("---")