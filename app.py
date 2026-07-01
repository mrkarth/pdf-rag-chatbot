#------Setup and import--------

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="PDF")
st.title("Chat with your PDF")

#cache so these only load once per session, not on every rerun

@st.cache_resource
def load_embeding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_chroma_collection():
    client = chromadb.Client() #in-memory vector DB, resets each session
    return client.get_or_create_collection(name="pdf_chunks")

embedder = load_embeding_model()
collection = get_chroma_collection()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

#----------PDF upload and chunking-----------

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text=""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks=[]
    start=0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap # move forward, but re-include the overlap
    return chunks

#------------Embedding and storing (the indexing phase)------------

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if "indexed_file" not in st.session_state or st.session_state.indexed_file != uploaded_file.name:
        with st.spinner("Reading and indexing your PDF..."):
            text = extract_text_from_pdf(uploaded_file)
            chunks = chunk_text(text)

            #clear any previous PDF's data
            existing_ids = collection.get()["ids"]
            if existing_ids:
                collection.delete(ids=existing_ids)
            
            embeddings = embedder.encode(chunks).tolist()
            ids = [f"chunk_{i}" for i in range(len(chunks))]

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks
            )

            st.session_state.indexed_file = uploaded_file.name
        st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}")

#-----------Query and answer (the retrieval + generation phase)------------

if uploaded_file is not None:
    question = st.text_input("Ask a question about your PDF")

    if question:
        with st.spinner("Thinking..."):
            question_embedding = embedder.encode([question]).tolist()

            results = collection.query(
                query_embeddings=question_embedding,
                n_results=3 # top 3 most relevant chunks 
            )
            retrived_chunks = results["documents"][0]
            context = "\n\n".join(retrived_chunks)

            prompt = f"""Answer the question using only the context below. If the answer isn't in the context, sayyou don't know.

Context:
{context}

Question: {question}

Answer: """
            
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("See retrieved context (what the AI actually read)"):
            for i, chunk in enumerate(retrived_chunks):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(chunk)

