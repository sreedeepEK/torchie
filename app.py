import time
import warnings
import gradio as gr
from dotenv import load_dotenv
from loader import load_embeddings
from langchain_groq import ChatGroq

load_dotenv()
vector_store = load_embeddings()

PROMPT_TEMPLATE = """
You are a friendly and knowledgeable assistant who is proficient in PyTorch. Answer any questions that are related to PyTorch and answer questions according to the user input. If you’re unsure of the answer, respond with "I'm not sure about that," without making up answers.   

After your conclusion, briefly summarize the entire response in a neat conversational way, making the user understand in under 200 lines.

User Query: {user_input}

Context:
{context}

Response:
"""


llm = ChatGroq(model='llama-3.1-8b-instant', temperature=1, max_retries=3)

# Chatbot response function
def chatbot_response(user_input, history):
    # Perform similarity search to find relevant context
    query_result = vector_store.similarity_search_with_score(query=user_input, k=1)
    if query_result:
        top_document = query_result[0][0].page_content

        # Format the conversation history and context
        conversation_context = "\n\n".join([f"User: {q}\nTorchie: {r}" for q, r in history])
        llm_query = PROMPT_TEMPLATE.format(user_input=user_input, context=f"{conversation_context}\n\n{top_document}")

        # Get the LLM's response
        start_time = time.time()
        llm_answer = llm.invoke(llm_query)
        elapsed_time = time.time() - start_time

        # Format the response
        response = f"{llm_answer.content}\n\nResponse time: {elapsed_time:.2f} seconds"
    else:
        response = "I'm sorry, I couldn't find any relevant information."

    # Return only the response
    return response, history 

# Gradio interface function
def interface(user_input, history):
    bot_response, history = chatbot_response(user_input, history) 
    return bot_response, history 

# Define Gradio UI components
iface = gr.Interface(
    fn=interface, 
    inputs=[
        gr.Textbox(label="Your Question:", placeholder="Type your PyTorch-related question here..."), 
        gr.State(value=[]) # State input to keep conversation history
    ],
    outputs=[
        gr.Textbox(label="Torchie Response", show_copy_button=True, interactive=True), 
        gr.State() # State output to hold and return the conversation history
    ],
    title="Torchie - Your PyTorch Assistant",
    description="Hi! I'm Torchie, your friendly PyTorch assistant. Ask me anything about PyTorch, and I'll do my best to help."
)

iface.launch()