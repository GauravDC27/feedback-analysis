# rag_v2/rag_chain.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from intent_classifier import classify_intent
from vector_store import query_vector_store
from visualization_generator import generate_chart
from context_manager import reframe_query, session_memory
import pandas as pd

class RAGState(TypedDict):
    query: str
    reframed_query: str
    intent: dict
    retrieved_docs: List[dict]
    response_messages: List[dict]

def node_context_manager(state: RAGState) -> RAGState:
    query = state["query"]
    history = session_memory.get("last_4_turns", [])

    # Reframe the query using memory
    reframed = reframe_query(query, history)
    state["reframed_query"] = reframed

    # Add current query to memory with placeholder bot response
    if len(history) >= 4:
        history.pop(0)
    history.append({"user": query, "bot": "[response placeholder]"})
    session_memory["last_4_turns"] = history

    return state

def node_classify_intent(state: RAGState) -> RAGState:
    state["intent"] = classify_intent(state["reframed_query"])
    return state

def node_retrieve(state: RAGState) -> RAGState:
    state["retrieved_docs"] = query_vector_store(state["reframed_query"], top_k=5)
    return state

def node_generate_outputs(state: RAGState) -> RAGState:
    df = pd.DataFrame(state["retrieved_docs"])
    intent = state["intent"]
    messages = []

    # Add text summary if applicable
    messages.append({"type": "text", "content": "Here are the results based on your query."})

    # Table
    if intent.get("needs_table"):
        html_table = df.to_html(index=False, classes="table table-striped", border=0)
        messages.append({"type": "table", "content": html_table})

    # Chart
    chart_type = intent.get("chart_type")
    if intent.get("needs_chart") and chart_type:
        x = intent.get("x_column", "Product")
        y = intent.get("y_column", "Score")
        agg = intent.get("aggregation")

        if x in df.columns and y in df.columns:
            if agg:
                df = df.groupby(x).agg({y: agg}).reset_index()
            chart_path = generate_chart(df, chart_type, x, y)
            messages.append({"type": "chart", "content": f"/{chart_path}"})
        else:
            messages.append({"type": "text", "content": f"Cannot create chart — missing columns: {x}, {y}."})

    # Save into session memory (final bot response to update context)
    session_memory["last_4_turns"][-1]["bot"] = messages[0]["content"]

    state["response_messages"] = messages
    return state

def build_rag_chain():
    builder = StateGraph(state_schema=RAGState)
    builder.set_entry_point("context")

    builder.add_node("context", node_context_manager)
    builder.add_node("classify_intent_node", node_classify_intent)
    builder.add_node("retrieve", node_retrieve)
    builder.add_node("generate_outputs", node_generate_outputs)

    builder.add_edge("context", "classify_intent_node")
    builder.add_edge("classify_intent_node", "retrieve")
    builder.add_edge("retrieve", "generate_outputs")
    builder.add_edge("generate_outputs", END)

    builder.set_finish_point("generate_outputs")
    return builder.compile()

rag_chain = build_rag_chain()
