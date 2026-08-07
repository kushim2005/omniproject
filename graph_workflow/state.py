from typing import TypedDict, List, Dict, Any, Annotated
import operator

class GraphState(TypedDict):
    """
    State object updated for Week 3 Self-RAG loop integration.
    """
    question: str
    route: str
    response: str
    documents: List[Dict[str, Any]]               # Raw retrieved documents
    filtered_documents: List[Dict[str, Any]]      # Graded relevant documents
    loop_count: int                              # Self-correction loop counter
    thought_process: Annotated[List[str], operator.add] # Log of steps taken