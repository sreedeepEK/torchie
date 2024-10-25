import torch
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader


device = "cuda" if torch.cuda.is_available() else "cpu"


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": device})

def read_docs_from_folder():
    
    loader = DirectoryLoader("./docs/", glob="**/*.txt", show_progress=True, loader_cls=lambda path: TextLoader(path, encoding='utf-8'))
    docs = loader.load()
    return docs

def create_and_save_faiss_index(docs, index_path="faiss_index"):
    
    texts = [doc.page_content for doc in docs]
    
    
    vector_store = FAISS.from_texts(texts, embedding_model)
    
  
    vector_store.save_local(folder_path=index_path)
    print(f"FAISS index saved at {index_path}")


documents = read_docs_from_folder()

create_and_save_faiss_index(documents)



def load_embeddings(index_path='faiss_index'):
    
    vector_store = FAISS.load_local(folder_path=index_path, embeddings=embedding_model)
    print("FAISS index loaded successfully")
    return vector_store