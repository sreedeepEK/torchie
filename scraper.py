import requests
from bs4 import BeautifulSoup

def extract_documentation(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the main documentation section
    documentation_section = soup.find('article', class_='pytorch-article')

    # Extract the text content
    documentation = documentation_section.get_text().strip()

    return documentation

# Replace with the actual URL of the PyTorch `torch.Tensor` documentation
url = "https://pytorch.org/docs/stable/tensors.html"

documentation = extract_documentation(url)

# Print the extracted documentation
print(documentation)