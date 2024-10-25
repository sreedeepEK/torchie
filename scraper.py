import os
import requests
from bs4 import BeautifulSoup

def extract_documentation(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract title 
    title = soup.title.string.strip()
    
    # Scrape main content
    documentation_section = soup.find('article', class_='pytorch-article')
    
    if documentation_section:
        documentation = documentation_section.get_text().strip()
    else:
        documentation = "No documentation found."

    # Extract internal links
    internal_links = []


    for link in soup.find_all('a', class_='reference internal', href=True):
        href = link['href']
        # Create a full URL for relative links
        full_url = f"https://pytorch.org/docs/stable/{href}"
        internal_links.append(full_url)

    return documentation, title, internal_links


# Save extracted content to a folder 
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



# specified url 

urls = []
for url in urls:
    documentation, title, internal_links = extract_documentation(url)
    save_text_to_folder(documentation, title)


for link in internal_links:
    documentation, title, _ = extract_documentation(link)
    save_text_to_folder(documentation, title)



#already scraped 

# https://pytorch.org/docs/stable/nn.html
# https://pytorch.org/docs/stable/nn.functional.html
# https://pytorch.org/docs/stable/torch.html
# https://pytorch.org/docs/stable/tensors.html
# https://pytorch.org/docs/stable/tensor_view.html 
# https://pytorch.org/docs/stable/autograd.html
# https://pytorch.org/docs/stable/cpu.html
# https://pytorch.org/docs/stable/cuda.html
# https://pytorch.org/docs/stable/utils.html