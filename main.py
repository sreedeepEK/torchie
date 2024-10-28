from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from loader import load_embeddings
from langchain_groq import ChatGroq
import time

app = FastAPI() 

vector_store = load_embeddings()

# Define the prompt template
PROMPT_TEMPLATE = """
You are a friendly and knowledgeable assistant with expertise in PyTorch and your name is Torchie. Answer any questions related to PyTorch in a simple, minimal, relaxed, cool conversational style. If you’re unsure of the answer, respond with "I'm not sure about that," without making up information.

User Query: {user_input}

Context:
{context}

Response:
""" 

llm = ChatGroq(model='llama-3.2-3b-preview', temperature=0.0, max_retries=2)

class Query(BaseModel):
    user_input: str
 
@app.post("/query")
async def get_response(query: Query):
    query_result = vector_store.similarity_search_with_score(query=query.user_input, k=5)
    
    if query_result:
        top_document = query_result[0][0].page_content
        llm_query = PROMPT_TEMPLATE.format(user_input=query.user_input, context=top_document)
        
        start_time = time.time()
        llm_answer = llm.invoke(llm_query)
        elapsed_time = time.time() - start_time
        
        return {"response": llm_answer.content, "time_taken": elapsed_time}
    else:
        return {"response": "No relevant documents found."} 
    
app.mount("/static", StaticFiles(directory="static"), name="static")



@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return f.read()