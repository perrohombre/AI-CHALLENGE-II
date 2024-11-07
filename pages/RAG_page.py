import PyPDF2
import re
import openai
import streamlit as st
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI
import requests
import os
from pathlib import Path
from dotenv import load_dotenv
from utils import read_answer

env_path = Path('..') / '.env'
load_dotenv(dotenv_path=env_path)

AZURE_EMBEDDING_OPENAI_ENDPOINT = os.getenv('AZURE_EMBEDDING_OPENAI_ENDPOINT')
AZURE_COMPLETION_OPENAI_ENDPOINT = os.getenv('AZURE_COMPLETIONS_OPENAI_ENDPOINT')
AZURE_EMBEDDING_API_KEY = os.getenv('AZURE_EMBEDDING_API_KEY')
AZURE_COMPLETION_API_KEY = os.getenv('AZURE_COMPLETIONS_API_KEY')

EMBEDDING_MODEL_NAME = "text-embedding-3-large"
COMPLETION_MODEL_NAME = "chatgpt4o"

embedding_client = AzureOpenAI(
    azure_endpoint=AZURE_EMBEDDING_OPENAI_ENDPOINT, 
    api_key=AZURE_EMBEDDING_API_KEY,  
    api_version="2024-06-01"
)

completions_client = AzureOpenAI(
    azure_endpoint=AZURE_COMPLETION_OPENAI_ENDPOINT, 
    api_key=AZURE_COMPLETION_API_KEY,  
    api_version="2024-06-01"
)

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def chunk_document_by_sections(text):
    text = text.strip()
    section_pattern = r'(\n\d+\.\s[A-Za-z].*)'
    sections = re.split(section_pattern, text)

    chunks = []
    if len(sections) > 1:
        for i in range(1, len(sections), 2):
            section_header = sections[i].strip()
            section_content = sections[i + 1].strip() if (i + 1) < len(sections) else ''
            chunks.append(f"{section_header}\n{section_content}")
    else:
        chunks.append(text)

    return chunks

def generate_embeddings(text_list):
    embeddings = []
    for text in text_list:
        embedding = embedding_client.embeddings.create(input=[text], model=EMBEDDING_MODEL_NAME).data[0].embedding
        embeddings.append(embedding)
    return embeddings

def save_embeddings_to_file(embeddings, filename="embeddings.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings_from_file(filename="embeddings.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

def find_similar_chunks(question_embedding, chunk_embeddings, chunks, top_k=5):
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_k_indices]

def generate_hypothetical_answer(question):
    messages = [
        {"role": "system", "content": "Jesteś pomocnym asystentem."},
        {"role": "user", "content": f"Provide a detailed answer to the following question in English:\n\n{question}\n\nAnswer"}
    ]

    response = completions_client.chat.completions.create(
        model=COMPLETION_MODEL_NAME,  
        messages=messages,
        max_tokens=256,
        temperature=0.7,
        n=1
    )

    return response.choices[0].message.content

def generate_final_answer(question, context):
    messages = [
        {"role": "system", "content": "Jesteś pomocnym asystentem."},
        {"role": "user", "content": f"Używając poniższego kontekstu, odpowiedz na pytanie w języku polskim:\n\nKontekst:\n{context}\n\nPytanie:\n{question}\n\nOdpowiedź:"}
    ]

    response = completions_client.chat.completions.create(
        model=COMPLETION_MODEL_NAME,
        messages=messages,
        max_tokens=256,
        temperature=0.7,
        n=1
    )

    return response.choices[0].message.content

def RAG_page():
    st.logo('assets/logo_black.png', icon_image='assets/logo_magenta.png')    
    st.header("RAG")

    progress_bar = st.progress(0)
    progress_bar.progress(5)

    if 'chunks' not in st.session_state:
        st.info("Przetwarzam dokument PDF...")
        pdf_path = "assets/Terms-and-Conditions.pdf"
        document_text = extract_text_from_pdf(pdf_path)
        chunks = chunk_document_by_sections(document_text)
        st.session_state['chunks'] = chunks
        progress_bar.progress(30)

        embeddings_file = "assets/embeddings.pkl"
        loaded_embeddings = load_embeddings_from_file(embeddings_file)

        if loaded_embeddings:
            st.info("Ładuję zapisane embeddingi...")
            chunk_embeddings = loaded_embeddings
            st.session_state['chunk_embeddings'] = chunk_embeddings
            progress_bar.progress(60)
            st.success("Embeddingi załadowane!")
        else:
            st.info("Generuję embeddingi dla fragmentów dokumentu...")
            chunk_embeddings = generate_embeddings(chunks)
            st.session_state['chunk_embeddings'] = chunk_embeddings
            save_embeddings_to_file(chunk_embeddings, embeddings_file)
            progress_bar.progress(60)
            st.success("Embeddingi wygenerowane i zapisane!")

    question = st.text_input("Zadaj pytanie dotyczące dokumentu:")

    if st.button("Uzyskaj odpowiedź") and question:
        with st.spinner("Generuję odpowiedź..."):
            progress_bar.progress(70)

            st.info("Generuję wstępną odpowiedź na podstawie pytania...")
            hypothetical_answer = generate_hypothetical_answer(question)

            st.info("Generuję embedding pytania...")
            question_embedding = generate_embeddings([hypothetical_answer])[0]
            progress_bar.progress(80)

            st.info("Znajduję najbardziej podobne fragmenty dokumentu...")
            similar_chunks = find_similar_chunks(
                question_embedding,
                st.session_state['chunk_embeddings'],
                st.session_state['chunks'],
                top_k=5
            )
            progress_bar.progress(90)

            context = "\n\n".join(similar_chunks)

            st.info("Generuję ostateczną odpowiedź...")
            final_answer = generate_final_answer(question, context)
            progress_bar.progress(100)

            st.subheader("Odpowiedź")
            st.write(final_answer)
            
            st.subheader("Istotne sekcje")
            for idx, chunk in enumerate(similar_chunks[:3]):
                with st.expander(f"Sekcja {idx+1}"):
                    st.write(chunk)

            st.success("Przetwarzanie zakończone!")
            read_answer(final_answer)
            

if __name__ == "__main__":
    RAG_page()