from app import app
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from tools import search_tool, wiki_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool


load_dotenv()


class ResearchResponse(BaseModel):
    topic: str = Field(description="The main topic of the user's query")
    summary: str = Field(description="A clear, empathetic explanation of the claims process")
    sources: list[str] = Field(description="A list of standard insurance principles or steps mentioned")


llm = ChatAnthropic(model='claude-haiku-4-5-20251001')
parser = PydanticOutputParser(pydantic_object=ResearchResponse)


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={'k': 3})


ciin_lookup_tool = create_retriever_tool(
    retriever,
    name="insurance_policy_lookup",
    description="Search this tool to find official definitions, principles, and rules from foundational CIIN insurance textbooks. Always use this before answering claims or underwriting questions."
)


tools = [search_tool, wiki_tool, ciin_lookup_tool]


prompt = ChatPromptTemplate.from_messages([
    ('system', '''You are a friendly, capable and highly knowledgeable insurance claims assistant 
    for an Insurance Company.
    Your PRIMARY purpose is to help users with insurance-related topics such as:
    - How to submit and process insurance claims
    - Understanding insurance policies
    - Risk assessment and fraud detection
    - Claim status and next steps
    - General insurance guidance

    You CAN answer general questions if asked:
    - End your response by reminding the user of your main purpose only if the response from the user is not in any way related to insurance
    
    You should NEVER:
    - Make up information
    - Hallucinate
    - Approve or deny claims yourself
    
    Always be professional, clear and empathetic.
        '''),
    ('placeholder', '{chat_history}'),
    ('human', '{query}'),
    ('placeholder', '{agent_scratchpad}'),
])


agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbosity=True)


chat_history = []

print("Hi! I am your insurance claims assistant. Type 'exit' to quit.\n")


while True:
    query = input("Reply: ")
    
    if query.lower() == 'exit':
        print("Goodbye! Stay insured!")
        break

    try:
        result = agent_executor.invoke({'query': query, 'chat_history': chat_history})
       
        response = result['output']

        if isinstance(response, list):
    # Extract just the text from the list
            for item in response:
                if isinstance(item, dict) and item.get('type') == 'text':
                    print(item['text'])
                else:
                    print(f"\nAssistant: {response}\n")

                chat_history.append(HumanMessage(content=query))
                chat_history.append(AIMessage(content=response))

    except Exception as e:
        print(f'\nError: {e}\n')



if __name__ == '__main__':
    print('Running in terminal mode...')
    chat_history = []
    while True:
        query = input('Reply:' )
        if query.lower() == 'exit': break
        result = agent_executor.invoke({'query': query, 'chat_history': chat_history})
        print(result['output'])