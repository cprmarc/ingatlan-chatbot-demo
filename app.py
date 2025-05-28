import subprocess
import sys

# Automatikus modultelepítés, ha hiányzik
def ensure_package_installed(package_name, import_name=None):
    try:
        __import__(import_name or package_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"{package_name} telepítve.")

# Szükséges csomagok ellenőrzése és telepítése
ensure_package_installed("langchain-openai", "langchain_openai")
ensure_package_installed("openai")
ensure_package_installed("langchain")
ensure_package_installed("langchain-community")
ensure_package_installed("faiss-cpu")
ensure_package_installed("streamlit")
ensure_package_installed("requests")
ensure_package_installed("beautifulsoup4")
ensure_package_installed("tiktoken")

# Importálások
import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Streamlit beállítások
st.set_page_config(page_title="Ingatlan Chatbot", page_icon="🏠")
st.title("🏠 Ingatlan Chatbot – Webes Tudásbázissal")

st.markdown("""
Ez a chatbot képes válaszolni az általad megadott ingatlanos témájú weboldalak tartalma alapján.  
Add meg a weboldal(ak) URL-jét, és kérdezz bármit!
""")

# URL-ek bekérése
url_list = st.text_area("🔗 Írd be az URL(eke)t, soronként egyet (pl. https://...):")

# Kérdés bekérése
user_question = st.text_input("❓ Kérdésed a megadott oldalakkal kapcsolatban:")

# Futtatás gombra
if st.button("💬 Válasz kérése") and url_list and user_question:
    with st.spinner("🔄 Betöltés és feldolgozás..."):
        try:
            # Weboldalak betöltése
            urls = url_list.strip().split("\n")
            loader = WebBaseLoader(urls)
            documents = loader.load()

            # Dokumentumdarabolás
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(documents)

            # Embedding + FAISS index
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)

            # Kérdés-válasz lánc
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(temperature=0),
                retriever=retriever,
                return_source_documents=True
            )

            result = qa_chain(user_question)

            st.subheader("💡 Válasz:")
            st.write(result['result'])

            st.subheader("📚 Forrás(ok):")
            for doc in result["source_documents"]:
                st.markdown(f"- [{doc.metadata['source']}]({doc.metadata['source']})")
        
        except Exception as e:
            st.error(f"Hiba történt: {str(e)}")
