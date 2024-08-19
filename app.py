# Importing necessary libraries and modules
# LangChain ke components ko use karne ke liye import karte hain
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

# Hugging Face ke dataset ko retrieve karne ke liye
from datasets import load_dataset

# CassIO engine jo Astra DB integration me use hota hai
import cassio

# PDF file se text nikalne ke liye PyPDF2 ka use karte hain
from PyPDF2 import PdfReader

# Streamlit ko import karte hain jo ki web app banane ke kaam aata hai
import streamlit as st

# Environment variables ko load karne ke liye
import os

# Streamlit page ko set karte hain
st.set_page_config(page_title="PDF Q&A Bot using AstraDB", layout="wide")

# Streamlit ka sidebar layout set karte hain
st.sidebar.title("PDF Q&A Bot")

# Environment variables ko load karte hain
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# PDF file ko read karte hain
pdfreader = PdfReader('budget_speech.pdf')

# PDF se text nikalte hain aur ek string me store karte hain
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# AstraDB ka session initialize karte hain
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# OpenAI LLM aur Embeddings ko initialize karte hain
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# AstraDB ke vector store ko initialize karte hain
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
    session=None,  # Session None set karte hain tak ki default session use ho
    keyspace=None,  # Keyspace None set karte hain tak ki default keyspace use ho
)

# Text ko split karne ke liye CharacterTextSplitter use karte hain
text_splitter = CharacterTextSplitter(
    separator = "\n",  # Line ke break par split karte hain
    chunk_size = 800,  # Chunk ka size 800 characters rakhte hain
    chunk_overlap  = 200,  # Chunk overlap 200 characters ka rakhte hain
    length_function = len,  # Length ko measure karne ke liye len() function use karte hain
)
texts = text_splitter.split_text(raw_text)

# Pehle 50 chunks ko AstraDB me store karte hain
astra_vector_store.add_texts(texts[:50])

# Streamlit sidebar par text display karte hain
st.sidebar.success(f"Inserted {len(texts[:50])} text chunks into AstraDB!")

# Vector Store Index Wrapper ko initialize karte hain
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# Streamlit page par Q&A section create karte hain
st.title("PDF Q&A Bot")
st.write("Aap apna sawaal neeche type karke submit karein aur bot aapko jawab dega.")

# Streamlit input field aur button ko initialize karte hain
question = st.text_input("Enter your question:", "")
submit_button = st.button("Submit")

# Agar user ne submit button click kiya
if submit_button and question:
    st.write(f"Question: {question}")

    # AstraDB se answer retrieve karte hain
    answer = astra_vector_index.query(question, llm=llm).strip()
    st.write(f"Answer: {answer}")

    # Relevant documents ko show karte hain
    st.write("Relevant Documents:")
    for doc, score in astra_vector_store.similarity_search_with_score(question, k=4):
        st.write(f"Score: {score}, Document: {doc.page_content[:84]}...")

# Agar user 'quit' type kare, toh loop break ho jaye
if question.lower() == "quit":
    st.stop()
