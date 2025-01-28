import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import warnings 
from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

warnings.filterwarnings('ignore')
 

device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": device})

def read_docs_from_folder():
    loader = DirectoryLoader("./docs/", glob="**/*.txt", show_progress=True, loader_cls=lambda path: TextLoader(path, encoding='utf-8'))
    docs = loader.load() 
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 50)
    
    chunks = []
    
    for doc in docs: 
        chunks.extend(text_splitter.split_documents([doc]))
    
    for i, chunk in enumerate(chunks[:5]):  # Adjust the range as needed
        print(f"Length of chunk {i}: {len(chunk.page_content)} characters")
            
    return chunks
    
def create_and_save_faiss_index(docs, index_path="faiss_index"):
    texts = [doc.page_content for doc in docs]
    vector_store = FAISS.from_texts(texts, embedding_model)
    vector_store.save_local(folder_path=index_path)
    print(f"FAISS index saved at {index_path}")


def load_embeddings(index_path='faiss_index'):
    vector_store = FAISS.load_local(folder_path=index_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
    print("FAISS index loaded successfully")
    return vector_store


if __name__ == "__main__":
    documents = read_docs_from_folder()
    create_and_save_faiss_index(documents)
