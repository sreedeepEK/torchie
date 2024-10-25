import time 
import torch 
import warnings
from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loader import load_embeddings  # Adjust the import according to your file structure

warnings.filterwarnings('ignore')
load_dotenv()

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-saved FAISS index (vector store)
vector_store = load_embeddings()  # Load the embeddings from the FAISS index

# Define a prompt template for the assistant
PROMPT_TEMPLATE = """
You are a helpful assistant who has great knowledge about PyTorch documentation.

User Query: {user_input}
Context: {context}

Please provide a detailed response based on the context.
"""

# Initialize the language model
llm = ChatGroq(model='llama-3.2-1b-preview', temperature=0.5, max_retries=2)

# Start the chat
print("\n")
print("Welcome to the PyTorch Documentation Chatbot!") 
print("Type 'exit' to quit the chat.")

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ['exit', 'quit', 'q']:
        print("Exiting the chat. Goodbye!")
        break
    
    # Perform similarity search
    query_result = vector_store.similarity_search_with_score(query=user_input, k=5)
    
    if query_result:
        # Extract the top relevant document
        top_document = query_result[0][0].page_content 
        
        # Format the query for the LLM with context
        llm_query = PROMPT_TEMPLATE.format(user_input=user_input, context=top_document)
        start_time = time.time()
        
        # Get response from the LLM
        llm_answer = llm.invoke(llm_query)
        print("Bot:", llm_answer.content)
        
        elapsed_time = time.time() - start_time 
        print("\n")
        print(f"Time taken: {elapsed_time:.2f} seconds")  
    else:
        print("No relevant documents found.")
