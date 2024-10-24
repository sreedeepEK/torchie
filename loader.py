from langchain_community.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

def read_docs_from_folder():
    loader = DirectoryLoader("./docs/", glob="**/*.txt", show_progress=True, loader_cls=lambda path: TextLoader(path, encoding='utf-8'))
    docs = loader.load()

    
    if docs:
        document = docs[0]  
        print(document.page_content[:500]) 
    else:
        print("No documents found")

# Call the function
read_docs_from_folder()
