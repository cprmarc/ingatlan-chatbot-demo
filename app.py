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

# Streamlit konfiguráció - MINDIG ELSŐ!
st.set_page_config(
    page_title="🏡 Ingatlan Asszisztens",
    page_icon="🏡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 🎨 Zenga.hu stílusú CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Alapértelmezett beállítások */
    .stApp {
        background: #ffffff;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        color: white;
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
        box-shadow: 0 2px 20px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        opacity: 0.8;
        margin-top: 0.5rem;
    }
    
    /* Container */
    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        font-size: 1rem;
        font-weight: 400;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1a1a1a;
        background: #ffffff;
        box-shadow: 0 4px 16px rgba(26,26,26,0.1);
    }
    
    /* Chat response styling */
    .chat-response {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 1px solid #e9ecef;
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06);
        position: relative;
    }
    
    .chat-response::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #1a1a1a 0%, #666666 100%);
        border-radius: 16px 16px 0 0;
    }
    
    .response-text {
        font-size: 1rem;
        line-height: 1.6;
        color: #2d2d2d;
        margin: 0;
    }
    
    /* Feature cards */
    .feature-card {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border-color: #1a1a1a;
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0.5rem 0;
    }
    
    .feature-desc {
        font-size: 0.9rem;
        color: #666666;
        line-height: 1.5;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    
    /* Buttons */
    .stButton > button {
        background: #1a1a1a;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #2d2d2d;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(26,26,26,0.2);
    }
    
    /* Success messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        color: #155724;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #1a1a1a !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-container {
            padding: 0 1rem;
        }
        
        .main-title {
            font-size: 2rem;
        }
        
        .chat-response {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# (st.set_page_config már meghívva fent)

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">🏡 Ingatlan Asszisztens</div>
    <div class="main-subtitle">Minden, ami ingatlan. Minden, amire szükséged van.</div>
</div>
""", unsafe_allow_html=True)

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
        return ""

# Adatok betöltése háttérben
if 'vectorstore' not in st.session_state:
    with st.spinner("Tudásbázis betöltése..."):
        for url in url_list:
            text = scrape_url(url)
            if text:
                chunks = text_splitter.split_text(text)
                all_docs.extend([Document(page_content=chunk) for chunk in chunks])
        
        # FAISS index
        faiss_index_path = "faiss_index"
        embedding = OpenAIEmbeddings()
        
        if os.path.exists(faiss_index_path):
            try:
                st.session_state.vectorstore = FAISS.load_local(faiss_index_path, embedding)
            except:
                st.session_state.vectorstore = None
        
        if st.session_state.vectorstore is None and all_docs:
            st.session_state.vectorstore = FAISS.from_documents(all_docs, embedding)
            st.session_state.vectorstore.save_local(faiss_index_path)

def get_answer(query):
    if len(query.split()) > 15:
        return "Kérlek, tegyél fel rövidebb kérdést (max 15 szó)."
    
    if 'vectorstore' not in st.session_state or st.session_state.vectorstore is None:
        return "A tudásbázis jelenleg nem elérhető."
    
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.get_relevant_documents(query)
    
    if not relevant_docs:
        return "Sajnos ebben a témában nem tudok segíteni. Próbálj meg ingatlanvásárlással, lakáshitellel vagy szerződésekkel kapcsolatos kérdést feltenni."
    
    prompt = f"Válaszolj a kérdésre szakszerűen, de érthetően, maximum 4-5 mondatban: {query}"
    chain = load_qa_chain(ChatOpenAI(temperature=0, max_tokens=200), chain_type="stuff")
    result = chain.run(input_documents=relevant_docs, question=prompt)
    return result

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Chat interface
st.markdown("### Tedd fel a kérdésed")
question = st.text_input("", 
                        placeholder="Például: Mi a lakáshitel folyamata? Mire figyelj ingatlannál?",
                        help="Kérdezz bármit az ingatlanvásárlásról, lakáshitelről vagy kapcsolódó témákról.")

if question:
    with st.spinner("Válasz készítése..."):
        response = get_answer(question)
    
    st.markdown(f"""
    <div class="chat-response">
        <div class="response-text">{response}</div>
    </div>
    """, unsafe_allow_html=True)

# Features section
st.markdown("### Miben tudok segíteni?")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🏠</div>
        <div class="feature-title">Ingatlanvásárlás</div>
        <div class="feature-desc">Lépésről lépésre végigvezetlek a vásárlás folyamatán</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📋</div>
        <div class="feature-title">Szerződések</div>
        <div class="feature-desc">Segítek megérteni a jogi dokumentumokat</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">💰</div>
        <div class="feature-title">Lakáshitel</div>
        <div class="feature-desc">Információk a finanszírozási lehetőségekről</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔑</div>
        <div class="feature-title">Birtokbaadás</div>
        <div class="feature-desc">Minden a kulcsátadás folyamatáról</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
