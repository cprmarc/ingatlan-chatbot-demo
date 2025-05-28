import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
import os
import glob

# 🔑 API kulcs (ehhez előbb regisztrálnod kell az https://platform.openai.com oldalon)
os.environ["OPENAI_API_KEY"] = "az_openai_kulcsod_ide"

# 📚 Tudásanyag betöltése mappából
document_dir = "tudasanyagok"  # ide dobhatod be a fájlokat
all_docs = []

for filepath in glob.glob(os.path.join(document_dir, "*.txt")):
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(content)
        all_docs.extend([Document(page_content=chunk) for chunk in chunks])

# ⚖️ Vektorizálás (embedding) és indexelés
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(all_docs, embedding)

# ✉️ Kérdésválaszoló rendszer
def get_answer(query):
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query)
    if not relevant_docs:
        return "Sajnos ebben nem tudok segíteni."

    chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")
    result = chain.run(input_documents=relevant_docs, question=query)
    return result

# 🌐 Streamlit felület
st.title("🏡 Ingatlan Chatbot Demo")
question = st.text_input("✍️ Tedd fel a kérdésed:")

if question:
    response = get_answer(question)
    st.write("\n\n**Válasz:**")
    st.write(response)
