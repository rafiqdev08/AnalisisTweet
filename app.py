import streamlit as st
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from PIL import Image

# Load model from .pkl file
with open('dataset.pkl', 'rb') as model_file:
    modelsvc_loaded = pickle.load(model_file)

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\W+', ' ', text)  # Remove symbols and non-alphanumeric characters
    text = text.lower()  # Convert text to lowercase
    stemmer = StemmerFactory().create_stemmer()  # Initialize stemmer
    text = stemmer.stem(text)  # Stemming
    return text


# Main page function
def main():
    st.sidebar.image('logo.png', use_column_width=True)
    st.sidebar.markdown('[Universitas XYZ](https://handayani.ac.id)')
    
    st.markdown("<h1 style='text-align: center; color: #50C878;'>Analisis Sentimen Tweet Pemilu 2024</h1>", unsafe_allow_html=True)

    # Input text from user
    userText = st.text_input('Masukkan Tweet:', placeholder='Paste tweet terkait Pemilu 2024 di sini...')
    
    # Button for sentiment analysis
                # Preprocess the text
                text_clean = preprocess_text(userText)

                # Transform text with the model and predict sentiment
                text_vector = modelsvc_loaded['vectorizer'].transform([text_clean])
                prediction_proba = modelsvc_loaded['classifier'].predict_proba(text_vector)
                
                # Determine sentiment label
                sentiment_label = 'positif' if proba_positif >= 0.5 else 'negatif'

                # Load corresponding image
                if sentiment_label == 'positif':
                    image = Image.open('./images/positive.png')
                else:
                    image = Image.open('./images/negative.png')
                image = image.resize((int(image.width / 2), int(image.height / 2)))

        else:
            st.warning('Masukkan teks untuk menganalisis.')

if __name__ == '__main__':
    main()
