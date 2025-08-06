import os
import re
import json
import hashlib
import fitz
import nltk
from concurrent.futures import ThreadPoolExecutor

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Gemini API
import google.generativeai as genai

# Streamlit
import streamlit as st

# -------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------
nltk.download("stopwords")

DATA_DIR = "data"
INDEX_PATH = "faiss_index"
os.makedirs(DATA_DIR, exist_ok=True)

# Read GEMINI API key from Streamlit secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found in Streamlit secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# -------------------------------------------------------------------------
# PDF Processing
# -------------------------------------------------------------------------
def load_pdf_fast(pdf_path):
    """Read PDF and extract all text."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def process_pdf(pdf_file):
    """Split PDF into chunks for embeddings."""
    pdf_path = os.path.join(DATA_DIR, pdf_file)
    try:
        text = load_pdf_fast(pdf_path)
        if not text.strip():
            return []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)
        return [Document(page_content=chunk) for chunk in chunks]
    except Exception as e:
        st.warning(f"Failed to process {pdf_file}: {e}")
        return []

def get_data_hash():
    """Generate MD5 hash of all PDFs for cache checking."""
    hash_md5 = hashlib.md5()
    for pdf_file in sorted(os.listdir(DATA_DIR)):
        if pdf_file.endswith(".pdf"):
            with open(os.path.join(DATA_DIR, pdf_file), "rb") as f:
                hash_md5.update(f.read())
    return hash_md5.hexdigest()

@st.cache_resource
def create_pdf_knowledge_base():
    """Create or load FAISS index from PDFs."""
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        return None

    current_hash = get_data_hash()
    if os.path.exists(INDEX_PATH) and os.path.exists("data_hash.txt"):
        with open("data_hash.txt", "r") as f:
            old_hash = f.read().strip()
        if old_hash == current_hash:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=GEMINI_API_KEY
            )
            return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    all_chunks = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_pdf, pdf_files)
        for chunks in results:
            all_chunks.extend(chunks)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=GEMINI_API_KEY
    )
    vector_store = FAISS.from_documents(all_chunks, embedding=embeddings)
    vector_store.save_local(INDEX_PATH)
    with open("data_hash.txt", "w") as f:
        f.write(current_hash)
    return vector_store

pdf_knowledge_base = create_pdf_knowledge_base()

# -------------------------------------------------------------------------
# FAQ Dataset
# -------------------------------------------------------------------------
def clean_text(text):
    """Clean text for TF-IDF matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def load_dataset(filepath):
    """Load FAQ dataset from JSON."""
    questions, answers = [], []
    if not os.path.exists(filepath):
        return questions, answers
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            q = item.get("q")
            a = item.get("Response")
            if q and a:
                questions.append(clean_text(q))
                answers.append(a)
    return questions, answers

DATASET_PATH = os.path.join(DATA_DIR, "train.json")
faq_questions, faq_answers = load_dataset(DATASET_PATH)

if faq_questions:
    vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"))
    faq_vectors = vectorizer.fit_transform(faq_questions)
else:
    vectorizer = None
    faq_vectors = None

def retrieve_faq_context(user_query):
    """Find the most relevant FAQ question-answer pair."""
    if faq_vectors is None:
        return None, None
    query_vec = vectorizer.transform([clean_text(user_query)])
    similarity = cosine_similarity(query_vec, faq_vectors)
    idx = similarity.argmax()
    score = similarity[0][idx]
    if score < 0.1:
        return None, None
    return faq_questions[idx], faq_answers[idx]

# -------------------------------------------------------------------------
# Gemini AI
# -------------------------------------------------------------------------
def get_gemini_response(user_query, mode="pdf"):
    """Generate AI response from PDFs or FAQ."""
    context = ""
    if mode == "pdf" and pdf_knowledge_base:
        docs = pdf_knowledge_base.similarity_search(user_query, k=3)
        context = "\n".join([d.page_content for d in docs])
    elif mode == "faq":
        context_q, context_a = retrieve_faq_context(user_query)
        if context_q and context_a:
            context = f"Reference Q: {context_q}\nReference A: {context_a}"

    if context:
        prompt = f"You are a helpful Indian legal assistant AI.\nContext:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    else:
        prompt = f"You are a helpful Indian legal assistant AI.\nQuestion: {user_query}\nAnswer:"

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating answer: {e}"

# -------------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------------
st.set_page_config(page_title="AI Legal Assistant", layout="wide")
st.title("AI Legal Assistant for Indian Law")
st.write("Ask legal questions based on Indian Constitution or IPC using uploaded documents or FAQ knowledge.")

mode = st.radio("Select mode:", ["pdf", "faq"], horizontal=True)
query = st.text_input("Enter your legal question:")

if st.button("Get Answer") and query:
    with st.spinner("Thinking..."):
        answer = get_gemini_response(query, mode)
    st.subheader("Answer:")
    st.write(answer)
