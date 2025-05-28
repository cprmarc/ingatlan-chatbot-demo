import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain

import os
import pickle
import requests
from bs4 import BeautifulSoup

# ------------------- Beállítások -------------------

url_list = [
    "https://tudastar.ingatlan.com/tippek/az-ingatlanvasarlas-menete/",
    "https://tudastar.ingatlan.com/tippek/tulajdonjog-fenntartashoz-kapcsolodo-vevoi-jog/",
    "https://tudastar.ingatlan.com/tippek/birtokbaadasi-jegyzokonyv-mire-valo-miert-jo-hogyan-toltsd-ki/",
    "https://bankmonitor.hu/lakashitel-igenyles/?gad_source=1&gad_campaignid=17136057347&gbraid=0AAAAACS7qzwhEL7nbt8ITPrnD-3gPjb4M&gclid=CjwKCAjw6NrBBhB6EiwAvnT_roXLVPgwNCTGYdGzCi2yuT7b7BcYuYGFvwI9SnR_IEq4ilAxURKBGhoCOP4QAvD_BwE"
]

cache_articles_path = "cache_articles.pkl"
cache_vectorstore_path = "faiss_vectorstore.pkl"

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=30)

# ------------------- Cikkek betöltése és cache-elése -------------------

def scrape_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = "\n".join(p.get_text() for p in paragraphs)
        return content
    except Exception as e:
        print(f"Hiba a {url} lekérésekor: {e}")
        return ""

if os.path.exists(cache_articles_path):
    with open(cache_articles_path, "rb") as f:
        all_texts = pickle.load(f)
else:
    all_texts = []
    for url in url_list:
        text = scrape_url(url)
        if text:
            all_texts.append(text)
    with open(cache_articles_path, "wb") as f:
        pickle.dump(all_texts, f)

# ------------------- Dokumentumok feldarabolása -------------------

all_docs = []
for text in all_texts:
    chunks = text_splitter.split_text(text)
    all_docs.extend([Document(page_content=chunk) for chunk in chunks])

# ------------------- Embedding és FAISS index -------------------

embedding = OpenAIEmbeddings()

if os.path.exists(cache_vectorstore_path):
    with open(cache_vectorstore_path, "rb") as f:
        vectorstore = pickle.load(f)
else:
    vectorstore = FAISS.from_documents(all_docs, embedding)
    with open(cache_vectorstore_path, "wb") as f:
        pickle.dump(vectorstore, f)

# ------------------- Kérdés-válasz függvény -------------------

def get_answer(query):
    if len(query.split()) > 10:
        return "Kérlek, max 10 szóból álló kérdést tegyél fel."

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.get_relevant_documents(query)
    if not relevant_docs:
        return "Sajnos ebben nem tudok segíteni."

    prompt = (
        "Válaszolj a kérdésre legfeljebb három mondatban, egyszerűen és tömören.\n\n"
        f"Kérdés: {query}\n"
        "Válasz:"
    )

    chain = load_qa_chain(ChatOpenAI(temperature=0, max_tokens=150), chain_type="stuff")
    result = chain.run(input_documents=relevant_docs, question=prompt)
    return result

# ------------------- Streamlit UI -------------------

st.title("🏡 Ingatlan Chatbot Demo (limitált)")

question = st.text_input("✍️ Tedd fel a kérdésed (max 10 szó):")

if question:
    response = get_answer(question)
    st.write("\n\n**Válasz:**")
    st.write(response)
