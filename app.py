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

# 🔗 Beégetett weboldalak
PREDEFINED_URLS = [
    "https://bankmonitor.hu/lakashitel-igenyles/",
    "https://tudastar.ingatlan.com/tippek/az-ingatlanvasarlas-menete/",
    "https://tudastar.ingatlan.com/tippek/tulajdonjog-fenntartashoz-kapcsolodo-vevoi-jog/" , 
    "https://www.zenga.hu/hello-otthon?headerid=clfuvspprbu3a0aw6xd8cir2p"
    "https://www.zenga.hu/hello-otthon?headerid=clfuvt6b9c1070bw8kismatfo"
]

# Streamlit beállítás
st.set_page_config(page_title="Ingatlan Chatbot", page_icon="🏠")
st.title("🏠 Ingatlan vásárlási aszisztens")
st.markdown("Gontdalan, páratlan, ingatlan.")

# Chat-előzmény
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Kérdés bekérése
user_question = st.chat_input("Írd be a kérdésed és nyomj Entert...")

# Ha van kérdés:
if user_question:
    with st.spinner("Gondolkodom a válaszon..."):
        try:
            # Első körben töltsük be és indexeljük a forrásokat
            if "vectorstore" not in st.session_state:
                loader = WebBaseLoader(PREDEFINED_URLS)
                documents = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = splitter.split_documents(documents)
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_documents(docs, embeddings)
                st.session_state.vectorstore = vectorstore

            # 🔍 Keresés a tudásbázisban
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            relevant_docs = retriever.get_relevant_documents(user_question)

            # 🔒 Ha nincs elég releváns dokumentum, ne válaszoljon
            if not relevant_docs:
                answer = "Ebben a témában sajnos nem tudok biztos válasszal szolgálni a rendelkezésre álló információk alapján."
            else:
                # Lánc létrehozása és válaszgenerálás
                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatOpenAI(temperature=0),
                    retriever=retriever,
                    return_source_documents=False
                )
                result = qa_chain(user_question)
                answer = result["result"]

            # Elmentjük a párbeszédet
            st.session_state.chat_history.append(("🧑", user_question))
            st.session_state.chat_history.append(("🤖", answer))

        except Exception as e:
            error_msg = f"Hiba történt: {str(e)}"
            st.session_state.chat_history.append(("🤖", error_msg))

# Párbeszéd megjelenítése
for speaker, text in st.session_state.chat_history:
    with st.chat_message(name=speaker):
        st.markdown(text)
