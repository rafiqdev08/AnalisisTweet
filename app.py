import streamlit as st
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from PIL import Image

# Load models from .pkl files
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

# Buzzer detection function
def detect_buzzer(text):
    # Implement your buzzer detection logic here
    # Example: return True if detected, else False
    pass

# Main page function
def main():
    st.sidebar.image('logo.png', use_column_width=True)
    st.sidebar.markdown('[Universitas XYZ](https://handayani.ac.id)')
    
    st.markdown("<h1 style='text-align: center; color: #50C878;'>Analisis Sentimen Tweet Pemilu 2024</h1>", unsafe_allow_html=True)

    # Input text from user
    userText = st.text_input('Masukkan Tweet:', placeholder='Paste tweet terkait Pemilu 2024 di sini...')
    
    # Checkbox for buzzer detection
    detect_buzzer_option = st.checkbox('Deteksi Buzzer')
    
    # Button for sentiment analysis
    if st.button('Analisis'):
        if userText:
            # Preprocess the text
            text_clean = preprocess_text(userText)

            # Transform text with the model and predict sentiment
            text_vector = modelsvc_loaded['vectorizer'].transform([text_clean])
            prediction_proba = modelsvc_loaded['classifier'].predict_proba(text_vector)
            
            # Get the probability of the positive class
            proba_positif = prediction_proba[0][1]
            
            # Determine sentiment label
            sentiment_label = 'positif' if proba_positif >= 0.5 else 'negatif'

            # Load corresponding image
            if sentiment_label == 'positif':
                image = Image.open('./images/positive.png')
            else:
                image = Image.open('./images/negative.png')
            image = image.resize((int(image.width / 2), int(image.height / 2)))

            # Display sentiment analysis results
            st.image(image, caption=sentiment_label)
            
            # Buzzer detection analysis
            if detect_buzzer_option:
                buzzer_detected = detect_buzzer(userText)
                st.write('Deteksi Buzzer:', 'Ya' if buzzer_detected else 'Tidak')
                
            st.button('Reload')  # Button for new analysis
        else:
            st.warning('Masukkan teks untuk menganalisis.')

if __name__ == '__main__':
    main()
