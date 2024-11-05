import time
import warnings 
warnings.filterwarnings('ignore')
import gradio as gr 
from dotenv import load_dotenv
from loader import load_embeddings  
from langchain_groq import ChatGroq


load_dotenv()
vector_store = load_embeddings()  


PROMPT_TEMPLATE = """
You are a friendly and knowledgeable assistant who is expert in PyTorch. Your name is Torchie. Answer any questions that is related to Pytorch and answer questions according to the user input. If you’re unsure of the answer, respond with "I'm not sure about that," without making up information.

User Query: {user_input}

Context:
{context}

Response:
"""

llm = ChatGroq(model='llama-3.2-3b-preview', temperature=0.0, max_retries=3)

def chatbot_response(user_input, history=None):
    if history is None:
        history = []  

    
    query_result = vector_store.similarity_search_with_score(query=user_input, k=5)
    if query_result:
        top_document = query_result[0][0].page_content
     
        conversation_context = "\n\n".join([f"User: {q}\nTorchie: {r}" for q, r in history])
        llm_query = PROMPT_TEMPLATE.format(user_input=user_input, context=f"{conversation_context}\n\n{top_document}")
        
        start_time = time.time()
        llm_answer = llm.invoke(llm_query)
        elapsed_time = time.time() - start_time
        
        response = f"{llm_answer.content}\n\nResponse time: {elapsed_time:.2f} seconds"
    else:
        response = "No relevant documents found."
    
    
    history.append((user_input, response))
    
    latest_interaction = f"**Torchie:** {response}"
    
    return latest_interaction, history 

# Gradio interface
iface = gr.Interface(
    fn=chatbot_response,
    inputs=[gr.Textbox(placeholder="Ask me anything about PyTorch", label="Type your question here"), gr.State()],
    outputs=[gr.Markdown(label="Response"), gr.State()],
    title="Torchie",
    description="Hi, I'm Torchie! Ask me anything about PyTorch, and I'll provide clear, insightful answers based on the documentation.",
    flagging_mode='never',
    examples=["What is nn.Conv2d", "Explain torch.autograd", "How to use torch.cuda for GPU acceleration?"],
    submit_btn="Submit",
    show_progress='full',
    theme="default",
    fill_width=True
)

iface.launch(share= True)