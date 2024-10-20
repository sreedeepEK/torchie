from phi.agent import Agent 
from phi.model.ollama import Ollama
from phi.tools.duckduckgo import DuckDuckGo 

web_agent = Agent(
    name="Web Agent",
    
    role="Search the web for information, explain context with good examples",
    
    model=Ollama(id="llama3.2:1b"),
    tools=[DuckDuckGo()],
    markdown=True
)
   
web_agent.print_response("what does torch.tensor do?",stream=True)
