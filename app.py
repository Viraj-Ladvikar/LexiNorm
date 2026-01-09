import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer,PorterStemmer,SnowballStemmer,LancasterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Download NLTK Dtaa 

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# Initilaizxe NlTJK tool

lemmatzer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()
snow_ball_stemmer = SnowballStemmer('english')
lancaster_stemmer = LancasterStemmer()
word_cloud = WordCloud()
stop_words = set(stopwords.words('english'))

st.set_page_config(
    page_title="LexiNorm",
    page_icon="app-logo.png",
    layout="wide"
)


with st.sidebar:
    col1, col2 = st.columns([1, 4])

    with col1:
        st.image("app-logo.png", width=40)

    with col2:
        st.markdown(
            "<h1 style='margin:0; padding:0'>LexiNorm</h1>"
            "<p style='margin-top:0;margin-bottom:0;padding:0; color:gray; font-size:13px;'>"
            "Text Normalization & WordCloud</p>",
            unsafe_allow_html=True
        )

    st.divider()
    tokenize_checkbox = st.checkbox("Tokenization")
    remove_stopwords_checkbox = st.checkbox("Remove stopwords")
    lemmatize_checkbox = st.checkbox("Lemmatization")
    stem_checkbox = st.checkbox("Stemming")
    
    if stem_checkbox:
        
        stem_selected_option = st.selectbox(
        "Select Stemming Method?",
        ("Porter Stemmer","SnowBall Stemmer","Lancaster Stemmer")
        )
        st.divider()
    generate_wordcloud_checkbox = st.checkbox("Generate WordCloud")

st.markdown(
            "<h1 style='margin:0; padding:0'>LexiNorm</h1>"
            "<p style='margin-top:0;margin-bottom:2px;padding:0; color:gray; font-size:13px;'>"
            "Text Normalization & WordCloud</p>",
            unsafe_allow_html=True
        )
input_text = st.text_area("Enter The Text in the input filed and choose the operation you want to perform",height=150)

tokens = word_tokenize(input_text) if input_text else []

# Tokenization
if tokenize_checkbox and input_text:
    st.subheader("Tokens")
    st.write(tokens)

# Remove StopWords
if remove_stopwords_checkbox and input_text:
    st.subheader("Removed Stopword")
    removed_word=[word for word in tokens if word.lower() not in stop_words]
    st.write(removed_word)       
        
# Lemmatization 

if lemmatize_checkbox and input_text:
    lemmatzer_token = [lemmatzer.lemmatize(word) for word in tokens]
    st.subheader("Lemmatized Text")
    st.write(lemmatzer_token)

# Stemming 

if stem_checkbox and input_text:
    if stem_selected_option =="Porter Stemmer":
        stemmed_token = [porter_stemmer.stem(word) for word in tokens]
    elif stem_selected_option == "Lancaster Stemmer":
        stemmed_token =[lancaster_stemmer.stem(word) for word in tokens]
    elif stem_selected_option == "SnowBall Stemmer":
        stemmed_token = [snow_ball_stemmer.stem(word) for word in tokens]
    st.subheader(f"Stemmed Text Using {stem_selected_option}")
    st.write(stemmed_token)

# Generate WordCloud 

if generate_wordcloud_checkbox and input_text:
    wordcloud = WordCloud(width=200, height=100, margin=2, background_color='black',colormap='Accent',mode='RGBA').generate(input_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig)