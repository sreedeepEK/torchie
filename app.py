import time
import warnings
warnings.filterwarnings('ignore')
import gradio as gr
from dotenv import load_dotenv
from loader import load_embeddings
from langchain_groq import ChatGroq

# Load environment variables and embeddings
load_dotenv()
vector_store = load_embeddings()

# Prompt template for the chatbot
PROMPT_TEMPLATE = """
You are a friendly and knowledgeable assistant who is proficent in PyTorch. Answer any questions that is related to Pytorch and answer questions according to the user input. If you’re unsure of the answer, respond with "I'm not sure about that," without making up answers.   

Emulate a deeply introspective, methodical reasoning style that emphasizes thorough exploration, self-questioning, and iterative analysis. Approach tasks with a stream-of-consciousness internal dialogue that breaks down complex thoughts into atomic steps, embracing uncertainty and continuous revision. The user has included the following content examples. 


After your conclusion, brief the entire summary in good neat conversational and make them understand.
 
User Query: {user_input}

Context:
{context}

Response:
"""

# Initialize the LLM
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
        response = "No relevant documents found."

    # Update the chat history
    history.append((user_input, response))

    # Return the latest interaction and updated history
    return history

# Gradio chat interface
with gr.Blocks() as demo:
    gr.Markdown("# Torchie - Your PyTorch Assistant")
    gr.Markdown("Hi, I'm Torchie! Ask me anything about PyTorch, and I'll provide clear, insightful answers based on the documentation.")

    # Chatbot interface
    chatbot = gr.Chatbot(label="Chat with Torchie")
    msg = gr.Textbox(label="Type your question here", placeholder="Ask me anything about PyTorch...")
    clear = gr.Button("Clear")

    # Function to handle user input
    def respond(message, chat_history):
        chat_history = chat_history or []
        bot_response = chatbot_response(message, chat_history)
        return bot_response

    # Connect the input and output to the chatbot
    msg.submit(respond, [msg, chatbot], chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the interface
demo.launch(share=True)