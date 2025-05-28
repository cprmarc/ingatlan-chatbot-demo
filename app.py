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

# Szövegtördelő beállítása
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=30)

all_docs = []

# 📚 Helyi fájlok betöltése (jelenleg kikommentelve, mert nincs fájl)
# document_dir = "tudasanyagok"  # ide dobhatod a .txt fájlokat
# import glob
# for filepath in glob.glob(os.path.join(document_dir, "*.txt")):
#     with open(filepath, "r", encoding="utf-8") as file:
#         content = file.read()
#         chunks = text_splitter.split_text(content)
#         all_docs.extend([Document(page_content=chunk) for chunk in chunks])

# 🌍 Online anyagok URL-jei - ide másold be a cikkek URL-jeit
url_list = [
    "https://tudastar.ingatlan.com/tippek/az-ingatlanvasarlas-menete/",
    "https://tudastar.ingatlan.com/tippek/tulajdonjog-fenntartashoz-kapcsolodo-vevoi-jog/",
    "https://tudastar.ingatlan.com/tippek/birtokbaadasi-jegyzokonyv-mire-valo-miert-jo-hogyan-toltsd-ki/",
    "https://bankmonitor.hu/lakashitel-igenyles",
]

def scrape_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = "\n".join(p.get_text() for p in paragraphs)
        return content
