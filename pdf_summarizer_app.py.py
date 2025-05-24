import streamlit as st
import pdfplumber
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import pandas as pd

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="PDF Summarizer", page_icon="📄")

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
    }
    h1 {
        color: #4CAF50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- TITLE ----------------------
st.markdown("<h1 style='text-align: center;'>📄 PDF Summarizer App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a PDF and get a smart summary in seconds.</p><hr>", unsafe_allow_html=True)

# ---------------------- SIDEBAR INPUTS ----------------------
st.sidebar.title("📁 Upload Your PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])
num_sentences = st.sidebar.slider("Summary length (in sentences)", 1, 10, 3)

# ---------------------- PROCESS PDF & SUMMARIZE ----------------------
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def generate_summary(text, n_sentences):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, n_sentences)
    return " ".join(str(sentence) for sentence in summary)

# ---------------------- MAIN APP LOGIC ----------------------
if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)
    st.subheader("🧾 Extracted Text:")
    st.text_area("Full Document Text", raw_text, height=200)

    if st.sidebar.button("Generate Summary"):
        st.subheader("📝 Summary:")
        summary_text = generate_summary(raw_text, num_sentences)
        st.write(summary_text)

        # Download summary
        st.download_button("📥 Download Summary", summary_text, file_name="summary.txt", mime="text/plain")
