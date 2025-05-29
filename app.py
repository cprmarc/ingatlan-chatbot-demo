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

# 🔗 Beégetett weboldalak (háttérben)
PREDEFINED_URLS = [
    "https://www.ingatlan.com/blog/lakasvasarlas-tippek",
    "https://www.ingatlan.com/blog/energetikai-tanusitvany",
    "https://www.ingatlan.com/blog/hitelkalkulator-mukodese"
]

# Streamlit oldalbeállítás
st.set_page_config(page_title="Ingatlan Chatbot", page_icon="🏠")
st.title("🏠 Ingatlan Chatbot – Tudásbázis Weboldalakról")
st.markdown("Írj be kérdést az alábbi mezőbe, a válaszokat a háttérben betöltött weboldalak alapján kapod.")

# 🔁 Inicializáljuk a memóriát a beszélgetéshez
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 💬 Felhasználói kérdés bekérése (ENTER küldi el)
user_question = st.chat_input("Írd be a kérdésed és nyomj Entert...")

# 💡 Betöltés és válaszgenerálás, ha érkezett kérdés
if user_question:
    with st.spinner("Gondolkodom a válaszon..."):
        try:
            # Tudásbázis betöltés (csak első kérdésnél)
            if "vectorstore" not in st.session_state:
                loader = WebBaseLoader(PREDEFINED_URLS)
                documents = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = splitter.split_documents(documents)
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_documents(docs, embeddings)
                st.session_state.vectorstore = vectorstore

            retriever = st.session_state.vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(temperature=0),
                retriever=retriever,
                return_source_documents=False  # 🔕 Nem kérünk forrásokat
            )

            result = qa_chain(user_question)
            answer = result['result']

            # 🔁 Elmentjük a párbeszédet
            st.session_state.chat_history.append(("🧑", user_question))
            st.session_state.chat_history.append(("🤖", answer))

        except Exception as e:
            error_msg = f"Hiba történt: {str(e)}"
            st.session_state.chat_history.append(("🤖", error_msg))

# 🗨️ Korábbi kérdés-válaszok megjelenítése
for speaker, text in st.session_state.chat_history:
    with st.chat_message(name=speaker):
        st.markdown(text)
