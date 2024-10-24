import time 
import torch 
import warnings
from pathlib import Path
from dotenv import load_dotenv
warnings.filterwarnings('ignore')
from langchain_groq import ChatGroq 
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings import create_and_save_embeddings, load_embeddings 
 

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_docs_from_folder(folder_path='/docs/'):
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

# Call the function to create and save embeddings
create_and_save_embeddings(documents, device=device)


# Load the embeddings
vector_store = load_embeddings()

# Define a prompt template
PROMPT_TEMPLATE = """
You are a helpful assistant who have great Knowledge about PyTorch documentation.

User Query: {user_input}
Context: {context}

Please provide a detailed response based on the context.
"""

# Parse LLM 
llm = ChatGroq(model='llama-3.2-1b-preview',
               temperature=0.5,
               max_retries=2)

print("\n")
print("Welcome to the PyTorch Documentation Chatbot!") 
print("Type 'exit' to quit the chat.")

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ['exit', 'quit', 'q']:
        print("Exiting the chat. Goodbye!")
        break
    
    query_result = vector_store.similarity_search_with_score(query=user_input, k=5)
    
    if query_result:
        top_document = query_result[0][0].page_content 

      
        llm_query = PROMPT_TEMPLATE.format(user_input=user_input, context=top_document)
        start_time = time.time()
        
        # Invoke the LLM
        llm_answer = llm.invoke(llm_query)
        print("Bot:", llm_answer.content)
        
        elapsed_time = time.time() - start_time 
        
        print("\n")
        print(f"Time taken: {elapsed_time:.2f} seconds")  
    else:
        print("No relevant documents found.")
        
        
        