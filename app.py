import streamlit as st
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import CharacterTextSplitter
import cassio
import os
from dotenv import load_dotenv

# Hinglish Comment: Ye code Streamlit web app banane ke liye hai jo Gemini model aur AstraDB ke saath integrate karta hai.

# Load environment variables
load_dotenv()

# Get GROQ API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if the API key is available
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not found. Please set it before running the application.")

# Initialize the Gemini model
try:
    model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
except Exception as e:
    raise RuntimeError(f"Failed to initialize ChatGroq model: {e}")

# Set up the prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# Initialize the parser
parser = StrOutputParser()

# Create the chain
chain = prompt_template | model | parser

# Initialize the vector store without embeddings
astra_vector_store = Cassandra(
    embedding=None,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None,
)

# Hinglish Comment: User ko PDF upload karne ka option dena.
st.title("Gemini-Powered QA System with PDF Upload")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Agar user ne PDF upload ki ho
if uploaded_file is not None:
    # Hinglish Comment: PDF file se text extract karna.
    pdfreader = PdfReader(uploaded_file)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    
    # Hinglish Comment: AstraDB ke saath connection establish karne ke liye CassIO use karte hain.
    cassio.init(token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"), database_id=os.getenv("ASTRA_DB_ID"))
    
    # Hinglish Comment: Text ko chhote chunks mein split karte hain takay ye Gemini model mein achhe se fit ho sake.
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    
    # Hinglish Comment: Chhote chunks ko AstraDB mein store karte hain.
    astra_vector_store.add_texts(texts[:50])
    
    st.success(f"PDF '{uploaded_file.name}' successfully processed and stored.")
    
    query_text = st.text_input("Enter your question:")
    
    # Agar user ne question diya ho
    if query_text:
        st.write(f"QUESTION: \"{query_text}\"")
        
        # Hinglish Comment: Gemini model ke through answer generate karte hain aur usse display karte hain.
        answer = astra_vector_index.query(query_text, llm=model).strip()
        st.write(f"ANSWER: \"{answer}\"")
        
        st.write("FIRST DOCUMENTS BY RELEVANCE:")
        for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
            st.write(f"[{score:.4f}] \"{doc.page_content[:84]} ...\"")
