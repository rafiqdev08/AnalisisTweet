import streamlit as st
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from PIL import Image
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')  # Unduh tokenizer untuk word_tokenize
nltk.download('stopwords')  # Jika Anda menggunakan stopwords


# Menambahkan link ke CDN Font Awesome
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
""", unsafe_allow_html=True)

# Menambahkan CSS untuk background image
st.markdown(
    """
    <style>
    .stApp {
        background: url('https://64.media.tumblr.com/83928a1806eea0bf918c0dff130c9dbc/tumblr_na39k233Xy1r83cqho1_400.gif');
        background-size: cover;
        background-position: center;
        z-index: -1;
    }
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.91);  /* Ubah nilai ini untuk mengatur tingkat transparansi */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Cache model dan data untuk mengurangi waktu loading
@st.cache_resource
def load_model():
    # with open('trainingdatamodel-8352.pkl', 'rb') as model_file:            #1st
    # with open('datatrainingmodel8352.pkl', 'rb') as model_file:             #2nd
    # with open('trainingdatamodel-8352-part1.pkl', 'rb') as model_file:      #3th
    # with open('dataset.pkl', 'rb') as model_file:                           #4th
    with open('trainingmodel-svm.pkl', 'rb') as model_file:                   #5th
        model = pickle.load(model_file)
    return model


@st.cache_data
def load_data():
    df = pd.read_csv('datasetlocation.csv')
    df['processed_text'] = df['full_text'].apply(preprocess_text_buzzer)
    return df


# Fungsi untuk preprocessing teks (untuk analisis sentimen)
def preprocess_text_sentiment(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s#]', '', text)
    text = text.lower()
    stemmer = StemmerFactory().create_stemmer()
    text = stemmer.stem(text)
    return text

# Fungsi untuk preprocessing teks (untuk deteksi buzzer)
def preprocess_text_buzzer(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s#]', '', text)
    words = word_tokenize(text)
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Fungsi untuk mendeteksi buzzer
def detect_buzzer(input_text, df):
    input_text_processed = preprocess_text_buzzer(input_text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
    input_vector = vectorizer.transform([input_text_processed])
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)
    max_similarity = similarity_scores.max()
    return max_similarity > 0.5

# Load model dan data
modelsvc_loaded = load_model()
df = load_data()

st.markdown("<h1 style='text-align: left; text-transform:uppercase; margin-top:-80px;'>Analisis Sentimen</h1>", unsafe_allow_html=True)

# Inisialisasi state untuk menyimpan input
if 'usertext' not in st.session_state:
    st.session_state.usertext = ""

# Ikon gambar kecil untuk menu
icon_home = Image.open('images/home_image.png')
icon_sentiment = Image.open('images/sentiment_icon.png')
icon_location = Image.open('images/location_icon.png')

# Tabs untuk navigasi menu
tabs = st.tabs(["Beranda", "Analisis Sentimen", "Tweet Map", "Visualisasi"])

with tabs[0]:
    st.markdown("<p style='text-align: left;'><i class='fa fa-home' style='color:#1D4FE1;'></i> Home</p>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; margin-top:-30px; text-transform:uppercase;'>Universitas Handayani<br> Makassar</h2>", unsafe_allow_html=True)
    st.image('logo.png', width=80)
    st.markdown("<h3 style='text-align: left;'>Selamat Datang!</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: left;'>ANALISIS SENTIMEN PERSEPSI PUBLIK TERHADAP HASIL PEMILU <br> BERBASIS MACHINE LEARNING MENGGUNAKAN ALGORITMA<br> SUPPORT VECTOR MACHINE (SVM) PADA APLIKASI TWITTER</h5>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown("<p style='text-align: left;'><i class='fa fa-comments' style='color:#1D4FE1;'></i> Tweet Sentiment Analysis</p>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; margin-top:-30px;'>Mulai Analisis ...</h2>", unsafe_allow_html=True)
    # st.markdown('<iframe src="https://x.com/search?q=pilpres%202024&src=typed_query?embed=true" style="height: 450px; width: 100%;"></iframe>', unsafe_allow_html=True)
    # st.markdown("""<blockquote class="twitter-tweet"><a href="https://twitter.com/TwitterDev/status/1354143047324299264"></a></blockquote><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>""", unsafe_allow_html=True)
    st.markdown('[Cari Pilpres 2024 di X](https://x.com/search?q=pilpres%202024&src=typed_query)', unsafe_allow_html=True)
    
    usertext = st.text_input("Input Status Tweet di sini!", placeholder='Paste status tweet di sini!', value=st.session_state.usertext)
    
    if st.button('Analisis'):
        if usertext.strip() == "":
            st.error("Input tidak boleh kosong. Silakan masukkan status tweet.")
        else:
            with st.spinner('Sedang menganalisis, harap tunggu...'):
                st.markdown(f"<h4>Status Tweet</h4><p>{usertext}</p>", unsafe_allow_html=True)

                # Preprocess input text
                text_clean = preprocess_text_sentiment(usertext)
                
                st.info(f"*Status tweet setelah diproses :*\n\n{text_clean}")

                # Transform text using the loaded vectorizer
                text_vector = modelsvc_loaded['vectorizer'].transform([text_clean])
                
                # Melakukan prediksi dengan probabilitas
                # prediction_proba = modelsvc_loaded['classifier'].predict_proba(text_vector)
                # sentiment_label = 'positif' if prediction_proba[0][1] >= 0.5 else 'negatif'

                # Menggunakan decision_function pada SVM
                sentiment_prediction = modelsvc_loaded['classifier'].decision_function(text_vector)
                sentiment_label = 'positif' if sentiment_prediction[0] > 1 else 'negatif'

                
                st.markdown(f"<h2>Hasil Analisis Sentimen</h2>", unsafe_allow_html=True)

                if sentiment_label == 'positif':
                    st.markdown(f"<h2 style='color: green;'><b>{sentiment_label.capitalize()}</b></h2>", unsafe_allow_html=True)
                    st.success('Analisis Sentimen Positif berhasil diproses!')
                    iconp = Image.open('images/wordcloud-positif.png')
                    st.image(iconp)
                else:
                    st.markdown(f"<h2 style='color: red;'><b>{sentiment_label.capitalize()}</b></h2>", unsafe_allow_html=True)
                    st.warning('Analisis Sentimen Negatif berhasil diproses!')
                    iconn = Image.open('images/wordcloud-negatif.png')
                    st.image(iconn)

                st.markdown(f"<h2>Deteksi Buzzer</h2>", unsafe_allow_html=True)

                is_buzzer = detect_buzzer(usertext, df)
                buzzer_status = "status tweet terindikasi berasal dari buzzer" if is_buzzer else "status tweet tidak terindikasi berasal dari buzzer"
                
                icon1 = Image.open('buzzer_icon.png')
                icon2 = Image.open('safe.png')
                if is_buzzer:
                    st.image(icon1, width=50)
                    st.warning(f"Buzzer Detected: {buzzer_status.capitalize()}")
                else:
                    st.image(icon2, width=50)
                    st.success(f"Buzzer Not Detected: {buzzer_status.capitalize()}")
                    st.balloons()

                st.success('Proses analisis selesai!')

                if st.button('Refresh', key='refresh_button'):
                    st.session_state.usertext = ""
                    st.experimental_rerun()

with tabs[2]:
    st.markdown("<p style='text-align: left;'><i class='fa fa-map' style='color:#1D4FE1 ;'></i>  Tweet Map Location</p>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; margin-top:-30px;'>Peta Lokasi Tweet</h2>", unsafe_allow_html=True)
    locations = df[['latitude', 'longitude']]
    st.map(locations)
    st.markdown("<p style='text-align: left;'><i class='fa fa-map' style='color:#1D4FE1 ;'></i>  Sistem mendeteksi lokasi tweet berdasarkan lokasi <i>tweet</i> dari dataset.</p>", unsafe_allow_html=True)


    location_name = df[['location']]
    # st.line_chart(location_name)
    # st.map(location_name)

with tabs[3]:
    
    st.markdown("<h2 style='text-align: left; margin-top:-30px;'>Visualisasi</h2>", unsafe_allow_html=True)
    
    # Tabs untuk navigasi menu
    tabs1 = st.tabs(["Wordcloud", "Confusion Matrix", "Hyperplane", "ROC"])
    with tabs1[0]:
        st.markdown("<p style='text-align: left;'><i class='fa fa-area-chart' style='color:#1D4FE1 ;'></i>  Wordcloud Visualisation</p>", unsafe_allow_html=True)
        st.image('images/wordcloud.png')
        st.markdown("<p style='text-align: left;'><i class='fa fa-area-chart' style='color:#1D4FE1 ;'></i>  Hasil analisis ditampilkan dalam bentuk <i>wordcloud</i></p>", unsafe_allow_html=True)
    with tabs1[1]:
        st.markdown("<p style='text-align: left;'><i class='fa fa-area-chart' style='color:#1D4FE1 ;'></i>  Wordcloud Visualisation</p>", unsafe_allow_html=True)
        st.image('images/confusion-matrix.png')
        st.markdown("<p style='text-align: left;'><i class='fa fa-area-chart' style='color:#1D4FE1 ;'></i>  Hasil analisis ditampilkan dalam bentuk grafik <i>Confusion Matrix</i>", unsafe_allow_html=True)
    with tabs1[2]:
        st.markdown("<p style='text-align: left;'><i class='fa fa-area-chart' style='color:#1D4FE1 ;'></i>  Wordcloud Visualisation</p>", unsafe_allow_html=True)
        st.image('images/hyperplane.png')
        st.markdown("<p style='text-align: left;'><i class='fa fa-area-chart' style='color:#1D4FE1 ;'></i>  Pada grafik ROC (Receiver Operating Characteristic) yang dihasilkan, terlihat bahwa model memiliki AUC (Area Under the Curve) sebesar 0.94, yang menunjukkan kinerja klasifikasi yang sangat baik. Nilai AUC ini mendekati 1.0, menandakan kemampuan model dalam membedakan antara kelas positif dan negatif dengan tingkat akurasi yang tinggi. Kurva ROC berada di atas garis diagonal acak, mengindikasikan bahwa model jauh lebih baik daripada prediksi acak. Dengan True Positive Rate (TPR) yang tinggi dan False Positive Rate (FPR) yang rendah, model mampu mengklasifikasikan sebagian besar data positif secara akurat sambil meminimalkan kesalahan prediksi untuk data negatif. Hasil ini memperkuat bahwa pendekatan machine learning yang digunakan efektif dalam analisis sentimen publik.</i>", unsafe_allow_html=True)
    with tabs1[3]:
        st.markdown("<p style='text-align: left;'><i class='fa fa-area-chart' style='color:#1D4FE1 ;'></i>  Wordcloud Visualisation</p>", unsafe_allow_html=True)
        st.image('images/roc.png')
        st.markdown("<p style='text-align: left;'><i class='fa fa-area-chart' style='color:#1D4FE1 ;'></i>  Hasil analisis ditampilkan dalam bentuk grafik <i>Confusion Matrix</i>", unsafe_allow_html=True)

st.markdown("<hr style='margin-top:0px;'><p style='text-align: center; margin-top:-20px;'>Dibuat dengan <i class='fa fa-heart' style='color:red;'></i> oleh Rafiq</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-top:-20px;'><em>© 2024 all rights reserved</em></p>", unsafe_allow_html=True)
