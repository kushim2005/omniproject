from langgraph.graph import StateGraph
from .state import GraphState
workflow = StateGraph(GraphState)
# Supervisor Node
def supervisor_node(state: GraphState):
    question = state["question"].lower()

    if "revenue" in question or "sql" in question:
        return {"route": "sql"}

    elif "chart" in question or "image" in question or "graph" in question:
        return {"route": "vision"}

    else:
        return {"route": "search"}

# Search Node
def search_node(state: GraphState):
    return {
        "response": "Search Agent Response"
    }

# SQL Node
def sql_node(state: GraphState):
    return {
        "response": "SQL Agent Response"
    }

# Vision Node
def vision_node(state: GraphState):
    return {
        "response": "Vision Agent Response"
    }
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("search", search_node)
workflow.add_node("sql", sql_node)
workflow.add_node("vision", vision_node)
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["route"],
    {
        "search": "search",
        "sql": "sql",
        "vision": "vision",
    },
)
workflow.add_edge("search", "__end__")
workflow.add_edge("sql", "__end__")
workflow.add_edge("vision", "__end__")
app_graph = workflow.compile()