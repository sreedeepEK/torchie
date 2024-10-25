import torch
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_and_save_embeddings(documents, filename='embeddings.pkl', device='cpu'):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": device})
    vector_store = FAISS.from_documents(documents, embeddings)
    
