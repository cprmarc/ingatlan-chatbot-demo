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

# 🎨 Custom CSS styling
st.markdown("""
<style>
    /* Háttér színek */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Főcím styling */
    .main-header {
        text-align: center;
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Input box styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.9);
        border: 2px solid #4CAF50;
        border-radius: 25px;
        padding: 15px 20px;
        font-size: 16px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Chat bubble válaszokhoz */
    .chat-message {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-left: 5px solid #4CAF50;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #4CAF50 0%, #45a049 100%);
    }
    
    /* Success/warning üzenetek */
    .stSuccess {
        background-color: rgba(76, 175, 80, 0.1);
        border: 1px solid #4CAF50;
        border-radius: 10px;
    }
    
    /* Spinner személyre szabás */
    .stSpinner > div {
        border-top-color: #4CAF50 !important;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(0,0,0,0.8);
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# 🔧 Streamlit konfiguráció
st.set_page_config(
    page_title="🏡 Ingatlan Asszisztens",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 📊 Sidebar konfiguráció
with st.sidebar:
    st.markdown("### ⚙️ Beállítások")
    
    # Színtéma választó
    theme = st.selectbox("🎨 Színtéma:", 
                        ["Zöld (alapértelmezett)", "Kék", "Narancs", "Lila"])
    
    # Válasz hossz beállítás
    max_words = st.slider("📝 Max szószám kérdésben:", 5, 20, 10)
    response_length = st.selectbox("📏 Válasz hossza:", 
                                  ["Rövid (3 mondat)", "Közepes (5 mondat)", "Részletes (8 mondat)"])
    
    st.markdown("---")
    st.markdown("### 📋 Mire tudok válaszolni?")
    st.markdown("""
    - 🏠 Ingatlanvásárlás menete
    - 💰 Lakáshitel információk  
    - 📋 Szerződések, jogok
    - 📝 Birtokbaadási jegyzőkönyv
    """)

# Dinamikus színtéma
color_schemes = {
    "Zöld (alapértelmezett)": "#4CAF50",
    "Kék": "#2196F3", 
    "Narancs": "#FF9800",
    "Lila": "#9C27B0"
}
primary_color = color_schemes[theme]

# Szövegtördelő beállítása
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=30)
all_docs = []

# URL lista
url_list = [
    "https://tudastar.ingatlan.com/tippek/az-ingatlanvasarlas-menete/",
    "https://tudastar.ingatlan.com/tippek/tulajdonjog-fenntartashoz-kapcsolodo-vevoi-jog/",
    "https://tudastar.ingatlan.com/tippek/birtokbaadasi-jegyzokonyv-mire-valo-miert-jo-hogyan-toltsd-ki/",
    "https://bankmonitor.hu/lakashitel-igenyles",
]

@st.cache_data
def scrape_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = "\n".join(p.get_text() for p in paragraphs)
        return content
    except Exception as e:
        st.warning(f"❌ Nem sikerült lekérni: {url}")
        return ""

# Adatok betöltése
with st.spinner("🔄 Tudásbázis betöltése..."):
    for url in url_list:
        text = scrape_url(url)
        if text:
            chunks = text_splitter.split_text(text)
            all_docs.extend([Document(page_content=chunk) for chunk in chunks])

# FAISS index
faiss_index_path = "faiss_index"
embedding = OpenAIEmbeddings()
vectorstore = None

if os.path.exists(faiss_index_path):
    try:
        vectorstore = FAISS.load_local(faiss_index_path, embedding)
        st.success("✅ Index betöltve!")
    except Exception as e:
        vectorstore = None

if vectorstore is None and all_docs:
    with st.spinner("🏗️ Index létrehozása..."):
        vectorstore = FAISS.from_documents(all_docs, embedding)
        vectorstore.save_local(faiss_index_path)
        st.success("✅ Index létrehozva!")

def get_answer(query, max_sentences=3):
    if len(query.split()) > max_words:
        return f"❌ Kérlek, max {max_words} szóból álló kérdést tegyél fel."
    
    if vectorstore is None:
        return "❌ A tudásbázis jelenleg nem elérhető."
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.get_relevant_documents(query)
    
    if not relevant_docs:
        return "🤷‍♂️ Sajnos ebben nem tudok segíteni."
    
    # Válasz hossz beállítása
    sentence_map = {
        "Rövid (3 mondat)": 3,
        "Közepes (5 mondat)": 5, 
        "Részletes (8 mondat)": 8
    }
    max_sentences = sentence_map[response_length]
    
    short_query = f"Válaszolj a kérdésre röviden, max {max_sentences} mondatban: " + query
    chain = load_qa_chain(ChatOpenAI(temperature=0, max_tokens=150), chain_type="stuff")
    result = chain.run(input_documents=relevant_docs, question=short_query)
    return result

# 🌐 Főoldal
st.markdown('<h1 class="main-header">🏡 Ingatlan Asszisztens</h1>', unsafe_allow_html=True)

# Központi chat interface
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("### 💬 Tedd fel a kérdésed!")
    question = st.text_input("", 
                           placeholder=f"Írj ide max {max_words} szót...",
                           help="Például: lakáshitel kamatok, birtokbaadás menete")
    
    if question:
        with st.spinner("🤔 Gondolkozom..."):
            response = get_answer(question)
        
        # Választ chat bubble-ben megjeleníteni
        st.markdown(f"""
        <div class="chat-message">
        <strong>🤖 Válasz:</strong><br>
        {response}
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Made with ❤️ using Streamlit | 🏡 Ingatlan Asszisztens v1.0
</div>
""", unsafe_allow_html=True)
