from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage,HumanMessage
from langgraph.graph.message import add_messages
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from langgraph.graph import END,  StateGraph, START


load_dotenv()

class AgentState(TypedDict):
    # The add_messages function tells LangGraph to append new messages to the list
    messages: Annotated[Sequence[BaseMessage], add_messages]

search_tool = TavilySearchResults(max_results=2)

def create_agent(llm, tools, system_prompt: str):
    """Helper to create an agent bound with specific tools."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    if tools:
        return prompt | llm.bind_tools(tools)
    return prompt | llm

llm = ChatGroq(api_key=os.environ.get("GROQ_API_KEY"),model="qwen/Qwen3.8-27B", temperature=0)

# 1. Researcher Agent Prompt
researcher_agent = create_agent(
    llm,
    tools=[search_tool],
    system_prompt="You are a brilliant web researcher. Search for accurate info. Once you have enough facts, clearly pass them on to the writer. Do not write the final article."
)

# 2. Writer Agent Prompt
writer_agent = create_agent(
    llm,
    tools=[],
    system_prompt="You are an expert technical writer. Take the research provided by the researcher and craft a compelling, structured summary. Keep it concise."
)

def researcher_node(state: AgentState):
    response = researcher_agent.invoke(state)
    # Ensure the graph knows which agent spoke by formatting the output name field
    if isinstance(response, AIMessage):
        response.name = "Researcher"
    return {"messages": [response]}

def writer_node(state: AgentState):
    response = writer_agent.invoke(state)
    if isinstance(response, AIMessage):
        response.name = "Writer"
    return {"messages": [response]}

# This prebuilt node automatically matches tool calls to the tools provided
tool_node = ToolNode(tools=[search_tool])

def route_researcher(state: AgentState):
    last_message = state["messages"][-1]
    # If the model requested tool calls, go execute them
    if last_message.tool_calls:
        return "call_tools"
    # Otherwise, pass the findings to the writer
    return "to_writer"

# 1. Initialize Graph with our State schema
workflow = StateGraph(AgentState)

# 2. Add all the Nodes
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Writer", writer_node)
workflow.add_node("tools", tool_node)

# 3. Connect the structural pathways (Edges)
workflow.add_edge(START, "Researcher")

# The researcher can either loop into tools or hand off to the writer
workflow.add_conditional_edges(
    "Researcher",
    route_researcher,
    {
        "call_tools": "tools",
        "to_writer": "Writer"
    }
)

# After tools run, the execution MUST go back to the researcher to process the results
workflow.add_edge("tools", "Researcher")

# Once the writer finishes its work, the workflow terminates
workflow.add_edge("Writer", END)

# 4. Compile the workflow
app = workflow.compile()

inputs = {"messages": [HumanMessage(content="Find the latest updates on NASA's Artemis program and write a 2-sentence summary.")]}

for output in app.stream(inputs, stream_mode="values"):
    # Print the last message added to the state sequentially
    last_msg = output["messages"][-1]
    print(f"[{getattr(last_msg, 'name', 'User')}]: {last_msg.content}\n")
    