import subprocess
import sys

# Automatikus modultelepítés
def ensure_package_installed(package_name, import_name=None):
    try:
        __import__(import_name or package_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"{package_name} telepítve.")

# Csomagok biztosítása
ensure_package_installed("langchain-openai", "langchain_openai")
ensure_package_installed("openai")
ensure_package_installed("langchain")
ensure_package_installed("langchain-community")
ensure_package_installed("faiss-cpu")
ensure_package_installed("streamlit")
ensure_package_installed("requests")
ensure_package_installed("beautifulsoup4")
ensure_package_installed("tiktoken")

# Importálás
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🔗 Beégetett weboldalak (ide írd be az URL-eket, amikből dolgozzon)
PREDEFINED_URLS = [
    "https://www.ingatlan.com/blog/lakasvasarlas-tippek",
    "https://www.ingatlan.com/blog/energetikai-tanusitvany",
    "https://www.ingatlan.com/blog/hitelkalkulator-mukodese"
    # ➕ Ide jöhetnek további oldalak
]

# Streamlit UI
st.set_page_config(page_title="Ingatlan Chatbot", page_icon="🏠")
st.title("🏠 Ingatlan Chatbot – Tudásbázis Weboldalakról")

st.markdown("Kérdezz bátran! A válaszokat a háttérben beállított szakmai weboldalak alapján kapod meg.")

# Kérdés bekérése
user_question = st.text_input("❓ Írd be a kérdésed az ingatlanokkal kapcsolatban:")

# Válaszgenerálás
if st.button("💬 Válasz kérése") and user_question:
    with st.spinner("🔄 Tudásbázis betöltése és válasz készítése..."):
        try:
            loader = WebBaseLoader(PREDEFINED_URLS)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)

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
