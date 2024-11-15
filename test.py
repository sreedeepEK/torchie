import time
import warnings 
warnings.filterwarnings('ignore')
import gradio as gr 
from dotenv import load_dotenv
from loader import load_embeddings  
from langchain_groq import ChatGroq 
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate 
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
vector_store = load_embeddings()  

model = ChatGroq(model='llama-3.1-70b-versatile', temperature=0.5, max_retries=3)

prompt = ChatPromptTemplate.from_messages([
    ('system', 'Your name is Torchie. You are a friendly and knowledgeable assistant who is excellent in PyTorch. Answer any questions that is related to Pytorch and answer questions according to the user input. Dont make up information.'),
    ('user', 'Question : {input}'),
])
# creating chain
chain = prompt | model | StrOutputParser()

# response = chain.invoke({'input': 'What does torch.nn do?'}) 
# print(response)


#reminding this dude to remember old convos!!
memory = ConversationBufferMemory()
chain = ConversationChain(
    llm=model,
    memory=memory
)



response = chain.invoke({'input': 'What does torch.nn do?'}) 
print(response)
 
 
