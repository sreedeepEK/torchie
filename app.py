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
You are a friendly and knowledgeable assistant with expertise in PyTorch, answering questions in a relaxed,cool, conversational style.
For technical questions, answer directly based on the given context. Avoid unnecessary preambles like "It seems like you're asking about..."
If you’re unsure of the answer, respond with "I'm not sure about that," without making up information.

User Query: {user_input}

Context (only for technical questions):
{context}

Response:
"""

llm = ChatGroq(model='llama-3.2-3b-preview', temperature=0.5, max_retries=2)

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
    outputs="text",
    title="PyTorch Documentation Chatbot",
    description="Ask questions about PyTorch, and I'll provide helpful answers based on the PyTorch documentation.",
    examples=["What is torch.autograd?", "How do I use nn.Module?", "Explain torch.nn.relu!"]
)


iface.launch(share=True)