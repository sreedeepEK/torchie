import os
import requests
from bs4 import BeautifulSoup

def extract_documentation(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    
    title = soup.title.string.strip()
   
    # scrap main content
    documentation_section = soup.find('article', class_='pytorch-article')
    documentation = documentation_section.get_text().strip()
    return documentation, title

def save_text_to_folder(documentation, title, folder_name='docs'):
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
   
    file_path = os.path.join(folder_name, f"{title}.txt")
   
    try:
        with open(file_path, 'w', encoding='utf-8') as text_file:
            text_file.write(documentation)
        print(f"Successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")
       

url = "https://pytorch.org/docs/stable/tensors.html"
documentation, title = extract_documentation(url)
save_text_to_folder(documentation, title)