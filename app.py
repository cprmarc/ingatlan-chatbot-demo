import os
import streamlit as st
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from bs4 import BeautifulSoup

# 📁 Beállítások
DATA_DIR = "data"
INDEX_FILE = "faiss_index"

# 🧠 OpenAI API-kulcs
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Kérlek, állíts be egy OpenAI API-kulcsot a .streamlit/secrets.toml fájlban vagy a környezeti változók között.")
    st.stop()

# 🧾 URL-ek, amiket be akarunk tölteni
URLS = [
    "https://ingatlan.com/tanacsok/lakasvasarlas",
    "https://ingatlan.com/tanacsok/energetikai-tanusitvany"
]

# 🧹 HTML szöveg kiszedése
def get_clean_text_from_html(content):
    soup = BeautifulSoup(content, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    return soup.get_text(separator=" ", strip=True)

# 🧠 Tudásbázis betöltése
def load_data():
    documents = []

    # 1️⃣ TXT fájlok
    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            if filename.endswith(".txt"):
                loader = TextLoader(os.path.join(DATA_DIR, filename), encoding='utf-8')
                documents.extend(loader.load())

    # 2️⃣ Webes források
    for url in URLS:
        loader = WebBaseLoader(url)
        raw_docs = loader.load()
        for doc in raw_docs:
            doc.page_content = get_clean_text_from_html(doc.page_content)
        documents.extend(raw_docs)

    return documents

# 🔄 Szöveg feldarabolása
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# 📦 Embedding és index készítése
def create_or_load_index(documents):
    i
