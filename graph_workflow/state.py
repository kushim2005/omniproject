from typing import TypedDict, List, Dict, Any, Annotated, Optional
import operator

class GraphState(TypedDict):
    """
    State object updated for Week 4 Langfuse observability integration.
    """
    question: str
    route: str
    response: str
    documents: List[Dict[str, Any]]               # Raw retrieved documents
    filtered_documents: List[Dict[str, Any]]      # Graded relevant documents
    loop_count: int                               # Self-correction loop counter
    thought_process: Annotated[List[str], operator.add]  # Log of steps taken
    trace_id: Optional[str]                       # Week 4: Langfuse trace ID
    session_id: Optional[str]                     # Week 4: Langfuse session ID