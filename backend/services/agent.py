import operator
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from backend.config import settings


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, researcher_model, summarizer_model, tools, checkpointer, system=""):
        self.system = system
        self.researcher = researcher_model.bind_tools(tools)
        self.summarizer = summarizer_model
        self.tools = {t.name: t for t in tools}

        graph = StateGraph(AgentState)
        graph.add_node("anthropic_researcher", self.call_anthropic)
        graph.add_node("action", self.take_action)
        graph.add_node("claude_summarizer", self.call_summarizer)

        graph.add_conditional_edges(
            "anthropic_researcher",
            self.exists_action,
            {True: "action", False: "claude_summarizer"},
        )
        graph.add_edge("action", "anthropic_researcher")
        graph.add_edge("claude_summarizer", END)
        graph.set_entry_point("anthropic_researcher")

        self.graph = graph.compile(checkpointer=checkpointer)

    def exists_action(self, state: AgentState) -> bool:
        return len(state["messages"][-1].tool_calls) > 0

    def call_anthropic(self, state: AgentState):
        print("🤖 CLAUDE IS RESEARCHING...")
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        return {"messages": [self.researcher.invoke(messages)]}

    def call_summarizer(self, state: AgentState):
        print("🧠 CLAUDE IS WRITING FINAL RESPONSE...")
        formatting_prompt = SystemMessage(content="""You are the final output generator for InsurAI Copilot.
Your job is simple: take whatever the researcher said and present it cleanly to the staff member.

Rules:
- FOLLOW-UPS: If the staff asks a specific question about a completed assessment, answer ONLY that question. DO NOT reprint the full report.
- If the researcher listed required fields, show that list clearly.
- If the researcher ran a NEW fraud assessment, present the full results clearly.
- NEVER say there is no research or no conversation history.
- NEVER make up context that was not in the conversation.
- Just present what was said, cleanly and professionally.""")

        nudge = HumanMessage(content=(
            "Please provide the final response to the staff member based on the conversation above. "
            "If the user asked a follow-up question, answer ONLY what was asked. "
            "Do not say there is no research or prior context — just deliver the response."
        ))
        return {"messages": [self.summarizer.invoke([formatting_prompt] + state["messages"] + [nudge])]}

    def take_action(self, state: AgentState):
        results = []
        for t in state["messages"][-1].tool_calls:
            print(f"🛠️  Calling Tool: {t['name']}")
            result = self.tools[t["name"]].invoke(t["args"])
            results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))
        print("🔙 Back to Claude!")
        return {"messages": results}


# ── LLM singleton ──────────────────────────────────────────────────────────
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# ── System prompts ─────────────────────────────────────────────────────────
MAIN_PROMPT = '''You are InsurAI Copilot, an AI assistant for insurance company staff.
[... keep your full prompt here unchanged ...]'''

FRAUD_CHAT_PROMPT = MAIN_PROMPT + """
FRAUD TOOL MODE — FIELD EXTRACTION ONLY:
You are operating as a claim data extraction assistant on the Fraud Assessment Tool page.
Your ONLY jobs are:
1. Extract claim field values from documents or descriptions the officer provides
2. Have a helpful conversation about the claim details
3. Always end your response with extracted fields in a JSON code block

CRITICAL RULES:
- D o NOT call fraud_detection_tool
- Do NOT run a fraud assessment
- Do NOT show fraud probability scores or risk levels

RESPONSE LENGTH RULES:
- Default to SHORT and concise — 3 to 5 sentences or a brief bullet list
- Only give long responses when the question explicitly asks for a full explanation
- Simple follow-up questions get 1 to 3 sentence answers only
- Never add "Key Points", "Bottom Line", or "Important Notes" padding sections
- Match your answer length to the complexity of the question

TOOL USAGE RULES:
- Use insurance_policy_lookup for: policy definitions, claims procedures, coverage questions, regulatory requirements
- Use search for: current news or real-time data only
- Do NOT use any tool for: simple follow-ups, clarifications, or questions answerable from conversation context already
- Call insurance_policy_lookup ONCE per question — never twice for the same question
"""