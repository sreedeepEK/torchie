import time 
import torch 
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')
from langchain_groq import ChatGroq 
from langchain.schema import Document
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings 
from scraper import extract_and_save_content, extract_function_links
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()


device = "cuda" if torch.cuda.is_available() else "cpu"
device 

from pathlib import Path

def read_docs_from_folder(folder_path='docs'):
    all_text = ""
    for file_path in Path(folder_path).rglob('*.txt'):
        with open(file_path, 'r', encoding='utf-8') as text_file:
            content = text_file.read()
            all_text += content 
    return all_text
  

documentation = read_docs_from_folder()
document = Document(page_content=documentation)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=200)
documents = text_splitter.split_documents([document]) 


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs = {"device": device})

vector_store = FAISS.from_documents(documents, embeddings)

#parse llm 
llm = ChatGroq(model='llama-3.2-1b-preview',
               temperature=0.0,
               max_retries=2)

print("\n")
print("Welcome to the PyTorch Documentation Chatbot!") 
print("Type 'exit' to quit the chat.")

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ['exit', 'quit','q']:
        print("Exiting the chat. Goodbye!")
        break
    
  
    query_result = vector_store.similarity_search_with_score(query=user_input, k=5)
    

    if query_result:
        top_document = query_result[0][0].page_content 


        llm_query = f"{user_input}\nContext:\n{top_document}"
        start_time = time.time()
        
        # Invoke the LLM
        llm_answer = llm.invoke(llm_query)
        print("Bot:", llm_answer.content)
        
        elapsed_time = time.time() - start_time 
        
        print("\n")
        print(f"Time taken: {elapsed_time:.2f} seconds")  
    else:
        pass 