# ============================================================
# OmniBrain — Week 3
# Member 1: Ravi
# Task: Self-RAG Logic (State Machine & Routing)
# ============================================================

from langgraph.graph import StateGraph, END
from .state import GraphState
from .self_rag_graders import SelfRAGGraders

# Initialize Graders
graders = SelfRAGGraders()

# ─── 1. CORE EXECUTION NODES ──────────────────────────────────

def supervisor_node(state: GraphState):
    question = state["question"].lower()
    thoughts = ["🕵️‍♂️ Supervisor: Analyzing query intent..."]
    
    # Reset or initialize state elements
    loop_count = state.get("loop_count", 0)
    
    if "revenue" in question or "sql" in question:
        route = "sql"
    elif "chart" in question or "image" in question or "graph" in question:
        route = "vision"
    else:
        route = "search"
        
    thoughts.append(f"🕵️‍♂️ Supervisor: Routing query to [{route}].")
    return {"route": route, "thought_process": thoughts, "loop_count": loop_count}


def search_node(state: GraphState):
    thoughts = ["🔍 Search Node: Retrieving document chunks..."]
    
    # Mocking retrieved documents mapping
    mocked_docs = [
        {"doc_id": "chunk_01", "text": "Deep Reinforcement learning is a subfield combining Q-learning with deep networks.", "page": 4},
        {"doc_id": "chunk_02", "text": "Supervised learning models predict outputs based on historical labels.", "page": 2}
    ]
    thoughts.append(f"🔍 Search Node: Retrieved {len(mocked_docs)} chunks from FAISS.")
    return {"documents": mocked_docs, "thought_process": thoughts}


def sql_node(state: GraphState):
    thoughts = ["🗄️ SQL Node: Accessing structured database..."]
    mocked_docs = [{"doc_id": "sql_01", "text": "Database shows standard revenue growth of 12% in Q3.", "page": 1}]
    return {"documents": mocked_docs, "thought_process": thoughts}


def vision_node(state: GraphState):
    thoughts = ["🖼️ Vision Node: Accessing multimodal pipeline..."]
    mocked_docs = [{"doc_id": "img_01", "text": "Bar chart illustrating error metrics decreasing over training.", "page": 12}]
    return {"documents": mocked_docs, "thought_process": thoughts}


# ─── 2. SELF-RAG GRADING & WORKFLOW NODES (Member 1 - Ravi) ───

def grade_documents_node(state: GraphState):
    """
    Evaluates all retrieved documents in state['documents'] for relevance 
    and filters out irrelevant noise.
    """
    question = state["question"]
    docs = state.get("documents", [])
    thoughts = ["🕵️‍♂️ Doc Grader: Commencing relevance grading..."]
    
    filtered_docs = []
    for doc in docs:
        score = graders.grade_document_relevance(question, doc["text"])
        if score == "yes":
            filtered_docs.append(doc)
            thoughts.append(f"  -> Match Found: Document [{doc['doc_id']}] is RELEVANT.")
        else:
            thoughts.append(f"  -> Filtered Out: Document [{doc['doc_id']}] is IRRELEVANT.")
            
    thoughts.append(f"🕵️‍♂️ Doc Grader: Completed. {len(filtered_docs)}/{len(docs)} documents remain.")
    return {"filtered_documents": filtered_docs, "thought_process": thoughts}


def generate_answer_node(state: GraphState):
    """Generates response based on filtered relevant documents."""
    filtered_docs = state.get("filtered_documents", [])
    thoughts = ["🧩 Generator: Synthesizing final answer based on relevant documents..."]
    
    if not filtered_docs:
        response = "No relevant context was found to safely answer the question."
    else:
        context_str = " ".join([d["text"] for d in filtered_docs])
        # Simulated generator synthesis
        response = f"Based on document records: {context_str}"
        
    return {"response": response, "thought_process": thoughts}


# ─── 3. SUB-AGENT PLACEHOLDERS (For Member 2 & Member 3) ─────

def query_rewriter_node(state: GraphState):
    """
    [MEMBER 2 PLACEHOLDER]
    Rewrites query to optimize retrieval in case of low document relevance.
    """
    current_query = state["question"]
    loop_count = state.get("loop_count", 0) + 1
    thoughts = [f"🔄 Query Rewriter [M2]: Loop Count = {loop_count}. Optimizing query syntax..."]
    
    # Simulating simple keyword refinement
    new_query = f"reinforcement learning theory and applications"
    thoughts.append(f"🔄 Query Rewriter [M2]: Rewrote query to: '{new_query}'")
    
    return {"question": new_query, "loop_count": loop_count, "thought_process": thoughts}


def self_correct_node(state: GraphState):
    """
    [MEMBER 3 PLACEHOLDER]
    Corrects hallucinations or answers lacking utility.
    """
    thoughts = ["🛠️ Self-Correction [M3]: Adjusting generation criteria to remove hallucinations..."]
    return {"thought_process": thoughts}


# ─── 4. CONDITIONAL ROUTING PATHWAYS (Member 1 - Ravi) ────────

def route_retrieval(state: GraphState) -> str:
    """Routes based on supervisor decision."""
    return state["route"]

def route_after_grading(state: GraphState) -> str:
    """Decides if we proceed to answer generation or rewrite the query."""
    filtered = state.get("filtered_documents", [])
    if not filtered:
        return "rewrite"
    return "generate"

def route_after_generation(state: GraphState) -> str:
    """Evaluates answer for hallucinations and utility."""
    answer = state["response"]
    facts = [d["text"] for d in state.get("filtered_documents", [])]
    question = state["question"]
    loop_count = state.get("loop_count", 0)
    
    # Guard to prevent infinite routing loop (Max 3 retries)
    if loop_count >= 3:
        return "accept"
        
    # Check Hallucination
    hallucination_score = graders.grade_hallucination(facts, answer)
    if hallucination_score == "no": # Answer has hallucinations
        return "correct"
        
    # Check Answer Utility
    utility_score = graders.grade_answer_utility(question, answer)
    if utility_score == "no": # Answer is safe but doesn't answer question
        return "rewrite"
        
    return "accept"


# ─── 5. COMPILE STATE GRAPH ──────────────────────────────────

workflow = StateGraph(GraphState)

# Add all execution nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("search", search_node)
workflow.add_node("sql", sql_node)
workflow.add_node("vision", vision_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("generate_answer", generate_answer_node)
workflow.add_node("query_rewriter", query_rewriter_node)
workflow.add_node("self_correct", self_correct_node)

# Set Entry
workflow.set_entry_point("supervisor")

# Conditional Router Edges
workflow.add_conditional_edges(
    "supervisor",
    route_retrieval,
    {
        "search": "search",
        "sql": "sql",
        "vision": "vision",
    }
)

# Connect retrievers to Document Relevance Grader
workflow.add_edge("search", "grade_documents")
workflow.add_edge("sql", "grade_documents")
workflow.add_edge("vision", "grade_documents")

# Document Grader conditional paths
workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "generate": "generate_answer",
        "rewrite": "query_rewriter"
    }
)

# Generation Evaluator conditional paths
workflow.add_conditional_edges(
    "generate_answer",
    route_after_generation,
    {
        "accept": END,
        "rewrite": "query_rewriter",
        "correct": "self_correct"
    }
)

# Connect self-correction and rewriter loops
workflow.add_edge("self_correct", "generate_answer")
workflow.add_edge("query_rewriter", "supervisor")

app_graph = workflow.compile()