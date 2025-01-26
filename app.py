import time
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from dotenv import load_dotenv
from loader import load_embeddings
from langchain_groq import ChatGroq

# Load environment variables and embeddings
load_dotenv()
vector_store = load_embeddings()

# Prompt template for the chatbot
PROMPT_TEMPLATE = """
You are a friendly and knowledgeable assistant who is proficient in PyTorch. Answer any questions that is related to PyTorch and answer questions according to the user input. If you’re unsure of the answer, respond with "I'm not sure about that," without making up answers.   


After your conclusion, brief the entire summary in good neat conversational and make them understand under 200 lines.
 
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
        response = "I'm sorry, I couldn't find any relevant information."

    # Update the chat history
    history.append((user_input, response))

    # Return the latest interaction and updated history
    return history

# Streamlit chat interface
# Streamlit chat interface
st.title("Torchie - Your PyTorch Assistant")
st.write("Hi! I'm Torchie, your friendly PyTorch assistant. Ask me anything about PyTorch, and I'll do my best to help.")

# Initialize session state for chat history if not already present
if 'history' not in st.session_state:
    st.session_state['history'] = []

# User input
user_input = st.text_input("Your Question:", placeholder="Type your PyTorch-related question here...")

# Handle user input and display response
if st.button("Submit") and user_input:
    st.session_state['history'] = chatbot_response(user_input, st.session_state['history'])

# Display only the most recent answer (after the 2nd question)
if len(st.session_state['history']) > 1:  # After second question is asked
    # Get the most recent user query and bot response
    user_query, bot_response = st.session_state['history'][-1]
    st.write(f"**User:** {user_query}")
    st.write(f"**Torchie:** {bot_response}")

