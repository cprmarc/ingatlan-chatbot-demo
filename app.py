import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain

import os
import glob
import requests
from bs4 import BeautifulSoup

# 🔑 API kulcs (OpenAI API key)
os.environ["OPENAI_API_KEY"] = "az_openai_kulcsod_ide"

# 📚 Tudásanyag betöltése helyi fájlokból
document_dir = "tudasanyagok"  # ide dobhatod a .txt fájlokat
all_docs = []

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)

for filepath in glob.glob(os.path.join(document_dir, "*.txt")):
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
        chunks = text_splitter.split_text(content)
        all_docs.extend([Document(page_content=chunk) for chunk in chunks])

# 🌍 Online anyagok URL-jei (szerkeszthető lista)
url_list = [
    # Írd ide a hasznos ingatlanos cikkek URL-jeit, pl:
    "https://www.penzcentrum.hu/otthon/ingatlanvasarlas-tanacsok-2025-01-01",
]

# Weboldal szöveg lekérése és tisztítása
def scrape_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = "\n".join(p.get_text() for p in paragraphs)
        return content
    except Exception as e:
        return ""

# Online tartalmak betöltése
for url in url_list:
    text = scrape_url(url)
    if text:
        chunks = text_splitter.split_text(text)
        all_docs.extend([Document(page_content=chunk) for chunk in chunks])

# ⚖️ Vektorizálás (embedding) és indexelés
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(all_docs, embedding)

# ✉️ Kérdés-válaszoló rendszer
def get_answer(query):
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query)
    if not relevant_docs:
        return "Sajnos ebben nem tudok segíteni."

    chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")
    result = chain.run(input_documents=relevant_docs, question=query)
    return result

# 🌐 Streamlit felület
st.title("🏡 Ingatlan Chatbot Demo")
question = st.text_input("✍️ Tedd fel a kérdésed:")

if question:
    response = get_answer(question)
    st.write("\n\n**Válasz:**")
    st.write(response)
