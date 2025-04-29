from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

from retriever import guest_info_tool
from langchain_openai import ChatOpenAI
from tools import search_tool, weather_info_tool, hub_stats_tool
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(r"C:\Projects\RAG_PoC\agents_course_hf\.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if API key is loaded
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

chat = ChatOpenAI(temperature=0)
tools = [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool]
chat_with_tools = chat.bind_tools(tools)
 
# AgentState
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }

# Graph
builder = StateGraph(AgentState)

# Nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition,)
builder.add_edge("tools", "assistant")
alfred = builder.compile()


# Usage
messages = [HumanMessage(content=input("Alfred: How can I help you?\nType here: "))]
response = alfred.invoke({"messages": messages})

print('-'*50 + "\nAlfred's Response:")
print(response['messages'][-1].content)