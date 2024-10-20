import requests 
import warnings
from bs4 import BeautifulSoup 
warnings.filterwarnings('ignore')
from langchain.schema import Document
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Fetch the webpage content
url = 'https://pytorch.org/docs/stable/tensors.html'
response = requests.get(url)

if response.status_code == 200:
    html_content = response.text

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove unwanted elements
    for class_name in ["pytorch-body", "pytorch-left-menu-search"]:
        for unwanted in soup.find_all(class_=class_name):
            unwanted.decompose()

    # Extract the title
    title = soup.title.string
    print("Title of the page:", title)

    # Get main content (targeting specific elements)
    content_elements = soup.select("h1, h2, h3, h4, p, ul, ol")  # Select headings and paragraphs
    all_text = "\n".join(element.get_text(separator=' ', strip=True) for element in content_elements)

    # Extract the links
    links = [a['href'] for a in soup.find_all('a', href=True)]

    # Create a Langchain Document
    document = Document(
        page_content=all_text, 
        metadata={
            'title': title,
            'links': links,
        }
    )
    
    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    documents = text_splitter.split_documents([document]) 
    
    # Create embeddings and vector database using FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Perform similarity search on FAISS vector store
    query_result = vector_store.similarity_search_with_score("What does torch.tensor do?", k=5)
    
    # Extract the most relevant document from the query result
    top_document = query_result[0][0].page_content  # The content of the top document

    # Initialize the Ollama LLM
    llm = OllamaLLM(model="llama3.2:1b")

    # Use the LLM to generate an answer based on the top document
    llm_query = f"Based on the following content, explain what torch.tensor does:\n\n{top_document}"
    
    llm_answer = llm.invoke(llm_query)
    print(llm_answer)