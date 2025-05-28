import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import requests
from bs4 import BeautifulSoup

# Beállítás
st.set_page_config(page_title="Ingatlanos Chatbot", page_icon="🏡")
st.title("🏡 Ingatlanos Web Chatbot")

# OpenAI API-kulcs
openai_api_key = st.sidebar.text_input("OpenAI API kulcs", type="password")

# Weboldalak listája
url_list = st.sidebar.text_area("Adj meg URL-eket (1 sor = 1 weboldal):", 
 [
    "https://tudastar.ingatlan.com/tippek/az-ingatlanvasarlas-menete/",
    "https://tudastar.ingatlan.com/tippek/tulajdonjog-fenntartashoz-kapcsolodo-vevoi-jog/",
    "https://tudastar.ingatlan.com/tippek/birtokbaadasi-jegyzokonyv-mire-valo-miert-jo-hogyan-toltsd-ki/",
    "https://bankmonitor.hu/lakashitel-igenyles",
]).splitlines()

# Gomb a chatbot betöltéséhez
if st.sidebar.button("🔄 Chatbot frissítése webes forrásokból"):

    if not openai_api_key:
        st.error("Kérlek, add meg az OpenAI API kulcsot!")
        st.stop()

    # URL-ekből szöveg lekérése
    def scrape_url(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            return soup.get_text()
        except Exception as e:
            st.warning(f"Hiba a(z) {url} betöltésekor: {e}")
            return ""

    # Szöveg chunkolása
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_docs = []

    for url in url_list:
        text = scrape_url(url)
        if text:
            chunks = text_splitter.split_text(text)
            all_docs.extend([Document(page_content=chunk, metadata={"source": url}) for chunk in chunks])

    if not all_docs:
        st.error("Nem sikerült szöveget betölteni egyik URL-ről sem.")
        st.stop()

    # Embedding + FAISS index
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(all_docs, embeddings)
    retriever = db.as_retriever()

    # Prompt sablon (kiegészíthető!)
    prompt_template = PromptTemplate.from_template("""
Válaszolj az alábbi kérdésre a dokumentumok alapján. 
Ha nem tudod a választ a webes szövegekből, akkor írd, hogy nem tudod.
Kérdés: {question}
""")

    # QA lánc definiálása
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    st.session_state.qa_chain = qa_chain
    st.success("✅ Chatbot betöltve a megadott weboldalakról!")

# Chatbox
if "qa_chain" in st.session_state:
    que
