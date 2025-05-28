import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

import os
import requests
from bs4 import BeautifulSoup
import pickle

# Szövegtördelő beállítása
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=30)

all_docs = []

# 📚 Helyi fájlok betöltése (jelenleg kikommentelve, mert nincs fájl)
# document_dir = "tudasanyagok"  # ide dobhatod a .txt fájlokat
# import glob
# for filepath in glob.glob(os.path.join(document_dir, "*.txt")):
#     with open(filepath, "r", encoding="utf-8") as file:
#         content = file.read()
#         chunks = text_splitter.split_text(content)
#         all_docs.extend([Document(page_content=chunk) for chunk in chunks])

# 🌍 Online anyagok URL-jei - ide másold be a cikkek URL-jeit
url_list = [
    "https://tudastar.ingatlan.com/tippek/az-ingatlanvasarlas-menete/",
    "https://tudastar.ingatlan.com/tippek/tulajdonjog-fenntartashoz-kapcsolodo-vevoi-jog/",
    "https://tudastar.ingatlan.com/tippek/birtokbaadasi-jegyzokonyv-mire-valo-miert-jo-hogyan-toltsd-ki/",
    "https://bankmonitor.hu/lakashitel-igenyles",
]

def scrape_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = "\n".join(p.get_text() for p in paragraphs)
        return content
    except Exception as e:
        st.warning(f"Nem sikerült lekérni az URL-t: {url}. Hiba: {e}")
        return ""

# Online cikkek feldolgozása, szöveg feldarabolása
for url in url_list:
    text = scrape_url(url)
    if text:
        chunks = text_splitter.split_text(text)
        all_docs.extend([Document(page_content=chunk) for chunk in chunks])

# 📁 Cache fájl elérési útja
cache_path = "faiss_vectorstore.pkl"

# Embedding példány
embedding = OpenAIEmbeddings()

vectorstore = None

# Vektor adatbázis betöltése vagy létrehozása
if os.path.exists(cache_path):
    try:
        with open(cache_path, "rb") as f:
            vectorstore = pickle.load(f)
    except Exception as e:
        st.warning(f"Cache betöltése sikertelen, újragenerálom az indexet. Hiba: {e}")
        vectorstore = None

if vectorstore is None:
    if not all_docs:
        st.error("Nincs betöltött dokumentum az index létrehozásához! Kérlek adj hozzá adatokat.")
    else:
        vectorstore = FAISS.from_documents(all_docs, embedding)
        with open(cache_path, "wb") as f:
            pickle.dump(vectorstore, f)

def get_answer(query):
    if len(query.split()) > 10:
        return "Kérlek, max 10 szóból álló kérdést tegyél fel."

    if vectorstore is None:
        return "A tudásbázis jelenleg nem elérhető."

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.get_relevant_documents(query)
    if not relevant_docs:
        return "Sajnos ebben nem tudok segíteni."

    short_query = "Válaszolj a kérdésre röviden, max 3 mondatban: " + query

    chain = load_qa_chain(ChatOpenAI(temperature=0, max_tokens=150), chain_type="stuff")
    result = chain.run(input_documents=relevant_docs, question=short_query)
    return result

# 🌐 Streamlit felület
st.title("🏡 Ingatlan Chatbot Demo (limitált)")

question = st.text_input("✍️ Tedd fel a kérdésed (max 10 szó):")

if question:
    response = get_answer(question)
    st.write("\n\n**Válasz:**")
    st.write(response)
