import requests
from bs4 import BeautifulSoup

def extract_documentation(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    #extract the title
    title = soup.title.string.strip()
    
    #main content
    documentation_section = soup.find('article', class_='pytorch-article')
    documentation = documentation_section.get_text().strip()

    return documentation

def save_to_text(documentation, filename='documentation.txt'):
    with open(filename, 'w',encoding='utf-8') as text_file:
        text_file.write(documentation)
        

url = "https://pytorch.org/docs/stable/tensors.html"
documentation = extract_documentation(url)
save_to_text(documentation)