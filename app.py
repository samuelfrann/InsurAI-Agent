import ast
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from tools import search_tool, wiki_tool

load_dotenv()


llm = ChatAnthropic(model='claude-haiku-4-5-20251001')
tools = [search_tool, wiki_tool]


prompt = ChatPromptTemplate.from_messages([
    ('system', '''You are a friendly and knowledgeable insurance claims assistant.
    Use your tools to research any topic. 
    Always end by offering help with insurance claims specifically.'''),
    ('placeholder', '{chat_history}'),
    ('human', '{query}'),
    ('placeholder', '{agent_scratchpad}'),
])


agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


app = FastAPI(title="Insurance Agent API")


class ChatRequest(BaseModel):
    query: str


@app.get('/')
async def serve_frontend():
    return FileResponse('index.html')

@app.get("/insurance_agent_icon.png")
async def serve_logo():
    return FileResponse("insurance_agent_icon.png")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Run the agent
        result = agent_executor.invoke({"query": request.query})
        raw_output = result["output"]

        # --- 3. CLEANUP LOGIC (The "Jargon" Filter) ---
        final_text = ""
        
        if isinstance(raw_output, str):
            clean_str = raw_output.strip()
            # Check for the "fake list" string: "[{'text': ..."
            if clean_str.startswith('[{'):
                real_list = ast.literal_eval(clean_str)
                final_text = real_list[0]['text']
            else:
                final_text = clean_str
        elif isinstance(raw_output, list):
            final_text = raw_output[0]['text']
        else:
            final_text = str(raw_output)

        return {"response": final_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 is required for Docker/AWS deployment
    uvicorn.run(app, host="0.0.0.0", port=8000)