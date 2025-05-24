import streamlit as st
import pdfplumber
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import spacy

# Download NLTK tokenizer
nltk.download("punkt")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Streamlit UI setup
st.set_page_config(page_title="📄 PDF Summarizer", layout="wide")
st.title("📄 PDF Summarizer App")
st.write("Upload a PDF, select page range and summary length, and get a clean summary.")

# Sidebar
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

start_page = st.sidebar.number_input("Start Page", min_value=1, value=1)
end_page = st.sidebar.number_input("End Page", min_value=start_page, value=start_page + 1)
num_sentences = st.sidebar.slider("Summary Length (Sentences)", 1, 20, 5)

# Extract text from selected pages
def extract_text(pdf_file, start, end):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        total = len(pdf.pages)
        if start > total:
            return "Invalid page range."
        for i in range(start - 1, min(end, total)):
            page = pdf.pages[i]
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

# Summarize using sumy (LSA)
def generate_summary(text, sentences):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences)
    return "\n".join(str(s) for s in summary)

# Extract keywords
def extract_keywords(text, max_keywords=10):
    doc = nlp(text)
    keywords = [chunk.text.lower().strip() for chunk in doc.noun_chunks]
    freq = {}
    for word in keywords:
        freq[word] = freq.get(word, 0) + 1
    sorted_kws = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [kw[0] for kw in sorted_kws[:max_keywords]]

# Main logic
if uploaded_file:
    with st.spinner("Processing..."):
        raw_text = extract_text(uploaded_file, start_page, end_page)
        word_count = len(raw_text.split())
        read_time = round(word_count / 200, 2)

        st.subheader("📄 Extracted Text")
        st.write(f"**Word Count:** {word_count} | **Reading Time:** {read_time} min")
        st.text_area("Full Text", raw_text, height=250)

        summary_text = generate_summary(raw_text, num_sentences)
        summary_word_count = len(summary_text.split())
        summary_time = round(summary_word_count / 200, 2)

        st.subheader("✂️ Summary")
        st.write(f"**Summary Word Count:** {summary_word_count} | **Reading Time:** {summary_time} min")
        st.text_area("Summary Output", summary_text, height=200)

        st.download_button("📥 Download Summary", summary_text, file_name="summary.txt")

        keywords = extract_keywords(summary_text)
        st.subheader("🔑 Key Concepts")
        st.write(", ".join(keywords))
