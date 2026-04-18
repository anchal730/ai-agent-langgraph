import os
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="AI Agent", page_icon="🤖")

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# =========================
# 🔐 Load Environment
# =========================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in .env")
    st.stop()

# =========================
# 🤖 Initialize LLM
# =========================
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# =========================
# 🛠️ Tools
# =========================
search = DuckDuckGoSearchRun()
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# =========================
# 🧠 State (UPDATED)
# =========================
class AgentState(TypedDict):
    input: str
    chat_history: List[str]
    tool: str
    tool_input: str
    output: str

# =========================
# 🤔 Step 1: Decide Tool / Answer
# =========================
def decide(state: AgentState):
    query = state["input"]
    history = state.get("chat_history", [])

    history_text = "\n".join(history)

    prompt = f"""
You are a helpful AI assistant.

Conversation so far:
{history_text}

User question:
{query}

Rules:
- If answer is in conversation → answer directly
- Otherwise choose tool:
    - search → latest/general info
    - wikipedia → factual info

Return:
1. Direct answer (normal text)
OR
2. JSON:
{{"tool": "...", "tool_input": "..."}}
"""

    response = llm.invoke(prompt).content

    import json
    try:
        data = json.loads(response)
        return {
            "tool": data["tool"],
            "tool_input": data["tool_input"],
            "chat_history": history
        }
    except:
        return {
            "output": response,
            "chat_history": history
        }

# =========================
# ⚡ Step 2: Execute Tool
# =========================
def act(state: AgentState):
    # if already answered from memory
    if "output" in state:
        return state

    tool = state["tool"]
    tool_input = state["tool_input"]
    history = state.get("chat_history", [])

    if tool == "search":
        result = search.run(tool_input)
    elif tool == "wikipedia":
        result = wiki.run(tool_input)
    else:
        result = "Invalid tool"

    return {
        "output": result,
        "chat_history": history
    }

# =========================
# 🔗 Build Graph
# =========================
graph = StateGraph(AgentState)

graph.add_node("decide", decide)
graph.add_node("act", act)

graph.set_entry_point("decide")
graph.add_edge("decide", "act")
graph.add_edge("act", END)

app = graph.compile()

# =========================
# 🎨 Streamlit UI
# =========================
st.title("🤖 AI Agent (LangGraph + Memory)")
st.write("Ask anything — now with memory 🧠")

# Chat UI history
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []

# Internal memory (for LLM context)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your question...")

if user_input:
    # display history
    st.session_state.chat_display.append(("user", user_input))

    with st.spinner("Thinking... 🤔"):
        try:
            result = app.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })

            output = result["output"]

        except Exception as e:
            output = f"❌ Error: {str(e)}"

    # store for display
    st.session_state.chat_display.append(("bot", output))

    # store for memory (important)
    st.session_state.chat_history.append(f"User: {user_input}")
    st.session_state.chat_history.append(f"Assistant: {output}")

# Display chat
for role, message in st.session_state.chat_display:
    if role == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)