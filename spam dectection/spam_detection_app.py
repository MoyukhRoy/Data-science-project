import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
df = pd.read_csv("C:\\Users\\user\\PycharmProjects\\pythonProject\\DIY Dataset\\spam.csv", encoding='latin-1')
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.rename(columns={'v1': 'message_type', 'v2': 'message'}, inplace=True)

# Preprocessing and feature extraction
df['message_type'] = df['message_type'].map({'ham': 0, 'spam': 1})

# Text preprocessing functions
nltk.download('punkt')
nltk.download('stopwords')
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

# Apply preprocessing to messages
df['processed_message'] = df['message'].apply(preprocess_text)

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['processed_message']).toarray()
y = df['message_type']

# Train the model
naive_bayes = MultinomialNB()
naive_bayes.fit(X, y)

# Streamlit web app
st.title("Spam Detection Web App")

user_input = st.text_area("Enter a message:")
processed_input = preprocess_text(user_input)
tfidf_input = tfidf_vectorizer.transform([processed_input]).toarray()

if st.button("Predict"):
    prediction = naive_bayes.predict(tfidf_input)
    if prediction[0] == 0:
        st.write("Prediction: Ham")
    else:
        st.write("Prediction: Spam")
