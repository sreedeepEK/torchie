import time 
import warnings
warnings.filterwarnings('ignore')
from langchain.schema import Document
from langchain_ollama import OllamaLLM
from scraper import extract_documentation, save_to_text
from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def read_docs(filename='documentation.txt'):
    with open(filename,'r',encoding='utf-8') as text_file:
        content = text_file.read()
        
    return content


documentation = read_docs()
document = Document(page_content=documentation)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
documents = text_splitter.split_documents([document]) 


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embeddings)


query_result = vector_store.similarity_search_with_score("What does torch.tensor do?", k=5)


top_document = query_result[0][0].page_content 

#parse llm 
llm = OllamaLLM(model="llama3.2:1b")

print("Welcome to the PyTorch Documentation Chatbot!") 
print("\n")
print("Type 'exit' to quit the chat.")

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ['exit', 'quit','q']:
        print("Exiting the chat. Goodbye!")
        break
    
  
    query_result = vector_store.similarity_search_with_score(user_input, k=5)
    

    if query_result:
        top_document = query_result[0][0].page_content 


        llm_query = f"{user_input}\nContext:\n{top_document}"
        start_time = time.time()
        
        # Invoke the LLM
        llm_answer = llm.invoke(llm_query)
        print("Bot:", llm_answer)
        
        elapsed_time = time.time() - start_time 
        
        print("\n")
        print(f"Time taken: {elapsed_time:.2f} seconds")  
    else:
        print("Bot: I couldn't find any relevant information.")