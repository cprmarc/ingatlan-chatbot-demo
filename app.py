import subprocess
import sys

# Automatikus modultelepítés
def ensure_package_installed(package_name, import_name=None):
    try:
        __import__(import_name or package_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"{package_name} telepítve.")

# Szükséges csomagok
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
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🔗 Előre beégetett weboldalak
PREDEFINED_URLS = [
    "https://tudastar.ingatlan.com/tippek/az-ingatlanvasarlas-menete/",
    "https://bankmonitor.hu/lakashitel-igenyles/",
    "https://www.zenga.hu/hasznos-tartalmak/ingatlanhitel-kalkulator-a-vasarlok-utmutatoja-a-hitelezes-vilagaban-clvqy5eaqlkyl06uyxws0mxf4",
    "https://tudastar.ingatlan.com/tippek/az-ingatlaneladas-folyamata/",
]

# 🌐 Egyéni weboldal betöltő
def load_custom_webpages(urls):
    documents = []
    for url in urls:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                # Weboldal szövegének kinyerése
                article_text = soup.get_text(separator="\n", strip=True)
                if len(article_text.strip()) > 100:
                    documents.append(Document(page_content=article_text, metadata={"source": url}))
            else:
                print(f"Hiba az URL betöltésekor: {url}")
        except Exception as e:
            print(f"Hiba történt a(z) {url} feldolgozásakor: {e}")
    return documents

# 🖼️ Streamlit UI beállítás
st.set_page_config(page_title="Ingatlan Chatbot", page_icon="🏠")
st.title("🏠 Ingatlan vásárlási aszisztens")
st.markdown("Gondtalan, páratlan, ingatlan.")

# 💬 Chat-előzmény tárolás
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 📥 Kérdés bekérése
user_question = st.chat_input("Írd be a kérdésed és nyomj Entert...")

# 🔍 Kérdés feldolgozása
if user_question:
    with st.spinner("Gondolkodom a válaszon..."):
        try:
            if "vectorstore" not in st.session_state:
                documents = load_custom_webpages(PREDEFINED_URLS)
                st.write(f"{len(documents)} dokumentum töltve be.")  # Debug: dokumentum számláló

                if not documents:
                    raise ValueError("Nem sikerült betölteni a dokumentumokat.")

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = splitter.split_documents(documents)
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_documents(docs, embeddings)
                st.session_state.vectorstore = vectorstore

            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            relevant_docs = retriever.get_relevant_documents(user_question)

            if not relevant_docs:
                answer = "Ebben a témában sajnos nem tudok biztos válasszal szolgálni a rendelkezésre álló információk alapján."
            else:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatOpenAI(temperature=0),
                    retriever=retriever,
                    return_source_documents=False
                )
                result = qa_chain(user_question)
                answer = result["result"]

            st.session_state.chat_history.append(("🧑", user_question))
            st.session_state.chat_history.append(("🤖", answer))

        except Exception as e:
            error_msg = f"Hiba történt: {str(e)}"
            st.session_state.chat_history.append(("🤖", error_msg))

# 💬 Párbeszéd megjelenítése
for speaker, text in st.session_state.chat_history:
    with st.chat_message(name=speaker):
        st.markdown(text)
