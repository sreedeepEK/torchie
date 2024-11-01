from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from loader import load_embeddings
from langchain_groq import ChatGroq
import time
from uuid import uuid4
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

vector_store = load_embeddings()

# Define the prompt template
PROMPT_TEMPLATE = """
You are Torchie, a friendly and knowledgeable AI assistant specializing in PyTorch. Your communication style is:
- Warm and approachable, using simple conversational language
- Patient and willing to break down complex concepts
- Precise with technical details while remaining accessible

Response Guidelines:
1. For greetings:
   - Respond naturally without mentioning PyTorch
   - Match the user's tone and enthusiasm
   - Keep it brief and friendly

2. For PyTorch-related questions:
   - Provide clear, structured explanations
   - Include practical code examples when relevant
   - Break down complex concepts into digestible parts
   - Add comments in code snippets to aid understanding

. When unsure:
   - Clearly state "I'm not sure about that"
   - Explain which parts you're uncertain about
   - Suggest reliable resources for further reading if applicable

Previous Conversation:
{chat_history}

Current Query: {user_input}

Additional Context:
{context}

Response:
"""

llm = ChatGroq(model='gemma2-9b-it', temperature=0.2, max_retries=2)

chat_histories = {}

class Query(BaseModel):
    user_input: str

@app.post("/query")
async def get_response(request: Request, query: Query, response: Response):
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid4())
        chat_histories[session_id] = []
        response.set_cookie(key="session_id", value=session_id)

    # Log the session ID
    logging.info(f"Session ID: {session_id}")

    chat_history = chat_histories.get(session_id, [])

    query_result = vector_store.similarity_search_with_score(query=query.user_input, k=5)

    if query_result:
        top_document = query_result[0][0].page_content

        formatted_history = "\n".join(chat_history[-5:])  # Limit to last 5 exchanges for brevity

        # Log the chat history and prompt
        logging.info(f"Chat History: {formatted_history}")
        logging.info(f"User Query: {query.user_input}")

        llm_query = PROMPT_TEMPLATE.format(user_input=query.user_input, context=top_document, chat_history=formatted_history)

        start_time = time.time()
        llm_answer = llm.invoke(llm_query)
        elapsed_time = time.time() - start_time

        chat_history.append(f"User: {query.user_input}")
        chat_history.append(f"Torchie: {llm_answer.content}")
        chat_histories[session_id] = chat_history

        return JSONResponse(content={"response": llm_answer.content, "time_taken": elapsed_time})
    else:
        return JSONResponse(content={"response": "No relevant documents found."})

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return f.read()