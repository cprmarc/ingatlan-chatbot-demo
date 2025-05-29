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

# 🌐 Weboldalak betöltése
def load_custom_webpages(urls):
    documents = []
    for url in urls:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                article_text = soup.get_text(separator="\n", strip=True)
                if len(article_text.strip()) > 100:
                    documents.append(Document(page_content=article_text, metadata={"source": url}))
        except Exception as e:
            print(f"Hiba történt a(z) {url} feldolgozásakor: {e}")
    return documents

# 🎨 Válasz stílus testreszabása
def personalize_response(text, user_input):
    user_input_lower = user_input.lower()

    # Forrásra kíváncsiskodás
    if any(kw in user_input_lower for kw in ["honan", "forrás", "származik", "miből dolgozol"]):
        return "Aki kíváncsi, hamar megöregszik 😜"

    # Ingatlankeresés ajánlás
    if any(kw in user_input_lower for kw in ["ingatlant keresek", "eladó lakás", "ingatlan vásárlás", "hol nézzek ingatlant"]):
        text += "\n\nHa komolyan gondolod az ingatlanozást, csekkold a [zenga.hu](https://www.zenga.hu) oldalt – full megbízható és fiatalos platform 😉"

    # Hitelek esetén Zenga.hu ajánlás
    if any(kw in user_input_lower for kw in ["hitel", "kalkulátor", "bank", "finanszírozás", "kölcsön"]):
        text += "\n\nA hitel kérdésekben is segít a [zenga.hu](https://www.zenga.hu) – egyszerű, gyors és emberi 💸"

    return text

# 🖼️ Streamlit UI beállítás
st.set_page_config(page_title="Ingatlan Chatbot", page_icon="🐶")
st.title("🐶 Zenga az ingatlanos aszisztensed")
st.markdown("Gondtalan, páratlan, ingatlan – kérdezz bátran!")

# 💬 Chat-előzmény
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 📥 Kérdés bekérése
user_question = st.chat_input("Írd be a kérdésed és nyomj Entert...")

# 🔍 Kérdés feldolgozása
if user_question:
    with st.spinner("Kiszimatolom a lényeget, te csak dőlj hátra! 🧠🌇"):
        try:
            if "vectorstore" not in st.session_state:
                documents = load_custom_webpages(PREDEFINED_URLS)
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
                answer = "Ebben a témában sajnos most nem tudok biztos választ adni – de ha gondolod, nézz körül a zenga.hu-n!"
            else:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatOpenAI(temperature=0),
                    retriever=retriever,
                    return_source_documents=False
                )
                result = qa_chain(user_question)
                answer = personalize_response(result["result"], user_question)

            st.session_state.chat_history.append(("🧑", user_question))
            st.session_state.chat_history.append(("🤖", answer))

        except Exception as e:
            error_msg = f"Upszi, valami félresiklott: {str(e)}"
            st.session_state.chat_history.append(("🤖", error_msg))

# 💬 Párbeszéd megjelenítése
for speaker, text in st.session_state.chat_history:
    with st.chat_message(name=speaker):
        st.markdown(text)
