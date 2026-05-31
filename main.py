import os
import uuid
import asyncio
import operator

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from rich.markdown import Markdown
from rich.console import Console
console = Console()

from langchain_core.messages import AnyMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_core.tools.retriever import create_retriever_tool
from tools import search_tool, wiki_tool, fraud_detection_tool, pdf_reader_tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_chroma import Chroma

os.environ['HF_HUB_OFFLINE'] = "1"
os.environ['TRANSFORMERS_OFFLINE'] = "1"


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={'k': 3})

ciin_lookup_tool = create_retriever_tool(
    retriever,
    name="insurance_policy_lookup",
    description="Search this tool to find official definitions, principles, and rules from foundational CIIN insurance textbooks. Always use this before answering claims or underwriting questions."
)


tools = [search_tool, wiki_tool, ciin_lookup_tool, fraud_detection_tool, pdf_reader_tool]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, researcher_model, summarizer_model, tools, checkpointer, system=''):
        self.system = system
        self.researcher = researcher_model.bind_tools(tools)
        self.summarizer = summarizer_model
        self.tools = {t.name: t for t in tools}

        graph = StateGraph(AgentState)
        graph.add_node('anthropic_researcher', self.call_anthropic)
        graph.add_node('action', self.take_action)
        graph.add_node('claude_summarizer', self.call_summarizer)

        graph.add_conditional_edges(
            'anthropic_researcher',
            self.exists_action,
            {True: 'action', False: 'claude_summarizer'}
        )
        graph.add_edge('action', 'anthropic_researcher')
        graph.add_edge('claude_summarizer', END)

        graph.set_entry_point('anthropic_researcher')

        self.graph = graph.compile(checkpointer=checkpointer)


    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0


    def call_anthropic(self, state: AgentState):
        print('🤖 CLAUDE IS RESEARCHING...')
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.researcher.invoke(messages)
        return {'messages': [message]}


    def call_summarizer(self, state: AgentState):
        print('🧠 CLAUDE IS WRITING FINAL RESPONSE...')
        messages = state['messages']
        formatting_prompt = SystemMessage(
            content='''You are the final output generator for InsurAI Copilot. 
            Your job is simple: take whatever the researcher said and present it cleanly to the staff member.

            Rules:
            - FOLLOW-UPS: If the staff asks a specific question about a completed assessment (e.g., "what is the probability?"), answer ONLY that question. DO NOT reprint the full claim details or previous reports.
            - If the researcher listed required fields, show that list clearly.
            - If the researcher ran a NEW fraud assessment, present the full results clearly.
            - If the researcher answered a question, present that answer clearly.
            - NEVER say there is no research or no conversation history.
            - NEVER say you need more information to generate a response.
            - NEVER make up context that was not in the conversation.
            - Just present what was said, cleanly and professionally.'''
        )

        nudge = HumanMessage(
            content="Please provide the final response to the staff member based on the conversation above. "
            "If the user asked a follow-up question, answer ONLY what was asked without reprinting the prior report. "
            "If the researcher already gave a complete answer, repeat it cleanly. "
            "Do not say there is no research or prior context — just deliver the response."
        )

        final_messages = [formatting_prompt] + messages + [nudge]
        message = self.summarizer.invoke(final_messages)
        return {'messages': [message]}


    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            if t['name'] == 'fraud_detection_tool':
                print("🔍 Running fraud detection model...")
            else:
                print(f"🛠️  Calling Tool: {t['name']}")
            
            result = self.tools[t['name']].invoke(t['args'])
            
            print(f"\n🚨 RAW TOOL OUTPUT: {result}\n") 
            
            results.append(ToolMessage(
                tool_call_id=t['id'],
                name=t['name'],
                content=str(result)
            ))
        print('🔙 Back to Claude!')
        return {'messages': results}


prompt = '''You are InsurAI Copilot, an AI assistant for insurance company staff.

Your PRIMARY purpose is to help users with insurance-related topics such as:
- How to submit and process insurance claims
- Understanding insurance policies
- Risk assessment and fraud detection
- Claim status and next steps
- General insurance guidance

Your tools:
- search: Search the web
- wikipedia: General knowledge lookup
- insurance_policy_lookup: Search official CIIN insurance textbooks
- fraud_detection_tool: ML model that assesses fraud risk on a claim

Always use the insurance_policy_lookup tool first before answering claims or underwriting questions.

FRAUD DETECTION RULES:

When staff says they want to run fraud detection OR provides claim details:

STEP 1 — If no claim data provided yet, respond with EXACTLY this list and nothing else:

"To run the fraud detection assessment, please provide the following 17 required fields:

1. Fault — 'Policy Holder' or 'Third Party'
2. BasePolicy — 'Liability', 'Collision', or 'All Perils'
3. VehicleCategory — 'Sport', 'Sedan', or 'Utility'
4. Month — Month of accident e.g. 'Jan', 'Dec'
5. Age — Claimant's age (number)
6. DayOfWeek — Day of accident e.g. 'Monday'
7. Year — Year of claim e.g. 1994
8. DayOfWeekClaimed — Day claim was filed e.g. 'Tuesday'
9. Make — Vehicle make e.g. 'Honda', 'Toyota'
10. AgeOfPolicyHolder — e.g. '26 to 30', '51 to 65'
11. NumberOfSuppliments — 'none', '1 to 2', '3 to 5', 'more than 5'
12. MonthClaimed — Month claim was filed e.g. 'Jul'
13. AgeOfVehicle — 'new', '3 years', 'more than 7'
14. PastNumberOfClaims — 'none', '1', '2 to 4', 'more than 4'
15. VehiclePrice — e.g. '20000 to 29000', 'more than 69000'
16. Sex — 'Male' or 'Female'
17. PoliceReportFiled — 'Yes' or 'No'

The following fields are optional — defaults will be used if not provided:
WeekOfMonth, WeekOfMonthClaimed, MaritalStatus, AccidentArea, Deductible,
DriverRating, Days_Policy_Accident, Days_Policy_Claim, WitnessPresent,
AgentType, AddressChange_Claim, NumberOfCars"

STEP 2 — When staff provides data, check which of the 17 required fields are missing.
If any are missing, ask for ONLY the missing ones in a short message.
Do NOT list all 17 again. Just name what is missing.

STEP 3 — When all 17 required fields are present, call fraud_detection_tool IMMEDIATELY.
Do not ask for confirmation. Do not summarise what you have. Just call it.

STEP 4 — After the tool runs, present the result clearly.
Always list the auto-filled fields if any were used.
Always remind staff that final decisions require human review.

FOLLOW-UP QUESTIONS: If staff asks a specific question about an already completed assessment (e.g., "what is the recommendation?" or "what was the probability?"), answer ONLY that specific question concisely. Do NOT reprint the entire fraud assessment report.

You CAN answer general questions if asked, but always end your response by reminding 
the user of your main purpose if the question is not insurance related.

You should NEVER:
- Make up information
- Hallucinate
- Approve or deny claims yourself

For all other questions, answer normally using the relevant tool.
Be direct and concise. Do not pad responses with unnecessary formatting.
'''


llm = ChatAnthropic(model='claude-sonnet-4-6')


async def main():
    DB_DIR = "./insurai_memory"
    DB_PATH = os.path.join(DB_DIR, 'insurai_memory')
    os.makedirs(DB_DIR, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as memory:
        bot = Agent(llm, llm, tools, checkpointer=memory, system=prompt)

        print("\n=== 🚀 INSURAI COPILOT ONLINE (STAFF MODE) ===")
        print("Type 'quit' or 'exit' to shut down.")
        print("Type 'clear' to wipe the agent's memory.\n")

        current_session_id = str(uuid.uuid4())

        while True:
            user_question = input("Insurer: ")

            if user_question.lower() in ['quit', 'exit']:
                print("Shutting down copilot...")
                break

            if user_question.lower() == 'clear':
                current_session_id = str(uuid.uuid4())
                print("\n🧹 Memory cleared! Starting a fresh conversation.\n")
                continue

            if not user_question.strip():
                continue

            messages = [HumanMessage(content=user_question)]
            config   = {'configurable': {'thread_id': current_session_id}}

            print('\n')
            full_response = ''

            async for event in bot.graph.astream_events({'messages': messages}, config):
                kind = event['event']
                node = event.get('metadata', {}).get('langgraph_node', '')
                if kind == 'on_chat_model_stream' and node == 'claude_summarizer':
                    content = event['data']['chunk'].content
                    if content:
                        full_response += content

            console.print(Markdown(full_response))
            print("\n" + "-" * 60 + "\n")


if __name__ == '__main__':
    asyncio.run(main())