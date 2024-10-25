import time
import torch
import warnings
import gradio as gr 
from dotenv import load_dotenv
from loader import load_embeddings  
from langchain_groq import ChatGroq
from langchain.schema import Document

load_dotenv()
vector_store = load_embeddings()  


PROMPT_TEMPLATE = """
You are a friendly and knowledgeable assistant with expertise in PyTorch,Answer any questions that is related to Pytorch and answer questions in simple, minimal, relaxed, cool conversational style. If you’re unsure of the answer, respond with "I'm not sure about that," without making up information.

User Query: {user_input}

Context:
{context}

Response:
"""

llm = ChatGroq(model='llama-3.2-90b-vision-preview', temperature=0.0, max_retries=3)

def chatbot_response(user_input):
    query_result = vector_store.similarity_search_with_score(query=user_input, k=5)
    
    if query_result:
     
        top_document = query_result[0][0].page_content
        llm_query = PROMPT_TEMPLATE.format(user_input=user_input, context=top_document)
        
        
        start_time = time.time()
        llm_answer = llm.invoke(llm_query)
        elapsed_time = time.time() - start_time
        
        return f"{llm_answer.content}\n\nTime taken: {elapsed_time:.2f} seconds"
    else:
        return "No relevant documents found."

iface = gr.Interface(
    fn=chatbot_response,
    inputs="text",
    outputs="markdown",
    title="Torchie",
    description="Hi, I'm Torchie! Ask me anything about PyTorch, and I'll provide clear, insightful answers based on the latest documentation.",
    flagging_mode='never',
        examples = ["What are tensors in Pytorch","How to use torch.cuda for GPU acceleration?","What is nn.Conv2d"],
    live=False,
    show_progress = 'full')



iface.launch(share= True)