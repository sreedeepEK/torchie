import requests 
from bs4 import BeautifulSoup 

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
             'text' : all_text}
    
    print(data) 
    
    