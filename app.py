import requests 
from bs4 import BeautifulSoup 
from langchain.schema import Document
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS 
from langchain.indexes import VectorstoreIndexCreator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

url = 'https://pytorch.org/docs/stable/torch.html#tensors'
response = requests.get(url)


if response.status_code == 200:
    html_content = response.text
    
    soup = BeautifulSoup(html_content, 'html.parser')
    soup.prettify()
    
    title = soup.title.string 
    #print(title)
    
    all_text = soup.get_text()
    #print(all_text)
    
    links = [a['href'] for a in soup.find_all('a', href=True)]
    #print('Links Found:', links)
    
    data  = { 
             'title' : title,
             'text' : all_text,
             'links' : links}


    #parse into Langchain
    document = Document(
        page_content=data['text'], 
        metadata = {
            'title' : data['title'],
            'links' : data['links'],
        }
    )

    
    
    #retrival [splitting the HTML data]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200, chunk_overlap = 20) 
    
    documents = text_splitter.split_documents([document]) 
     #print(f"Chunks created: {len(documents)}")
    
    
    #embedding & vector database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
     
    vector_store = FAISS.from_documents(documents, embeddings)
     
    
    index_creator = VectorstoreIndexCreator(embedding=embeddings)

    # Index the documents
    index = index_creator.from_documents(documents=documents)

    llm = OllamaLLM(model="llama3.2:1b")
    query_result = index.query("What is torch.nn?",llm=llm)
    
    print(query_result)