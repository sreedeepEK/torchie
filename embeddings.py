import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_and_save_embeddings(documents, filename='embeddings.pkl', device='cpu'):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": device})
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Save the vector store to a file
    with open(filename, 'wb') as f:
        pickle.dump(vector_store, f)
    print(f"Embeddings saved to {filename}")

def load_embeddings(filename='embeddings.pkl'):
    with open(filename, 'rb') as f:
        vector_store = pickle.load(f)
    return vector_store