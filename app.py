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
import pickle

# 📚 Tudásanyag betöltése helyi fájlokból
document_dir = "tudasanyagok"  # ide dobhatod a .txt fájlokat
all_docs = []

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=30)

for filepath in glob.glob(os.path.join(document_dir, "*.txt")):
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
        chunks = text_splitter.split_text(content)
        all_docs.extend([Document(page_content=chunk) for chunk in chunks])

# 🌍 Online anyagok URL-jei
url_list = [
    # Ide írd a hasznos cikkek URL-jeit
    "https://www.penzcentrum.hu/otthon/ingatlanvasarlas-tanacsok-2025-01-01",
]

def scrape_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = "\n".join(p.get_text() for p in paragraphs)
        return content
    except Exception:
        return ""

for url in url_list:
    text = scrape_url(url)
    if text:
        chunks = text_splitter.split_text(text)
        all_docs.extend([Document(page_content=chunk) for chunk in chunks])

# 📁 Embedding cache elérési út
cache_path = "faiss_vectorstore.pkl"

# ⚖️ Embedding létrehozása és indexelés (cache-elve)
embedding = OpenAIEmbeddings()

if os.path.exists(cache_path):
    with open(cache_path, "rb") as f:
        vectorstore = pickle.load(f)
else:
    vectorstore = FAISS.from_documents(all_docs, embedding)
    with open(cache_path, "wb") as f:
        pickle.dump(vectorstore, f)

# ✉️ Kérdés-válasz rendszer: max 5 releváns dokumentum, válasz max 3 mondat
def get_answer(query):
    if len(query.split()) > 10:
        return "Kérlek, max 10 szóból álló kérdést tegyél fel."

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.get_relevant_documents(query)
    if not relevant_docs:
        return "Sajnos ebben nem tudok segíteni."

    # Prompt, hogy max 3 mondat legyen a válasz
    prompt = (
        "Válaszolj a kérdésre legfeljebb három mondatban, egyszerűen és tömören.\n\n"
        "Kérdés: {question}\n"
        "Válasz:"
    )

    chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")
    result = chain.run(input_documents=relevant_docs, question=prompt.format(question=query))
    return result

# 🌐 Streamlit felület
st.title("🏡 Ingatlan Chatbot Demo (limitált)")

question = st.text_input("✍️ Tedd fel a kérdésed (max 10 szó):")

if question:
    response = get_answer(question)
    st.write("\n\n**Válasz:**")
    st.write(response)
