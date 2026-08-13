# ============================================================
# OmniBrain — Week 4
# Member 1: Vasu Sree
# Task: Langfuse Integration (Observability & Tracing)
# ============================================================

import os
import time
import uuid
from typing import Optional, Dict, Any
from functools import wraps

class LangfuseClient:
    """
    Centralized Langfuse observability client for OmniBrain.
    Wraps the Langfuse SDK with a graceful fallback to console
    logging when the SDK is not installed or keys are missing.
    """
    def __init__(self):
        self.enabled = False
        self.client = None
        self._current_trace = None
        self._spans = {}

        # Attempt to initialize Langfuse SDK
        try:
            from langfuse import Langfuse
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")
            host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

            if public_key and secret_key:
                self.client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host
                )
                self.enabled = True
                print("[INFO] Langfuse observability enabled.")
            else:
                print("[WARN] LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set. Tracing in console mode.")
        except ImportError:
            print("[WARN] langfuse package not installed. Running in console logging mode.")

    # ─── TRACE LIFECYCLE ──────────────────────────────────────

    def start_trace(self, name: str, input_data: Dict[str, Any], session_id: Optional[str] = None) -> str:
        """
        Start a new top-level trace for a user query.
        Returns a trace_id for linking spans.
        """
        trace_id = str(uuid.uuid4())

        if self.enabled:
            self._current_trace = self.client.trace(
                id=trace_id,
                name=name,
                input=input_data,
                session_id=session_id or trace_id,
                tags=["omnibrain", "week4", "rag-pipeline"]
            )
        else:
            print(f"\n[TRACE START] id={trace_id} name='{name}' input={list(input_data.keys())}")

        self._current_trace_id = trace_id
        return trace_id

    def end_trace(self, trace_id: str, output_data: Dict[str, Any], status: str = "success"):
        """Finalize a trace with its output and completion status."""
        if self.enabled and self._current_trace:
            self._current_trace.update(
                output=output_data,
                metadata={"status": status}
            )
        else:
            print(f"[TRACE END]   id={trace_id} status={status} output_keys={list(output_data.keys())}")

    # ─── SPAN MANAGEMENT ─────────────────────────────────────

    def start_span(self, trace_id: str, name: str, input_data: Dict[str, Any]) -> str:
        """
        Start a child span within a trace to track individual agent steps.
        Returns a span_id.
        """
        span_id = str(uuid.uuid4())
        start_time = time.time()

        if self.enabled and self._current_trace:
            span = self._current_trace.span(
                id=span_id,
                name=name,
                input=input_data,
                metadata={"start_time": start_time}
            )
            self._spans[span_id] = (span, start_time)
        else:
            print(f"  [SPAN START] id={span_id[:8]}... name='{name}'")
            self._spans[span_id] = (None, start_time)

        return span_id

    def end_span(self, span_id: str, output_data: Dict[str, Any], status: str = "success"):
        """End a child span and record its output and latency."""
        if span_id not in self._spans:
            return

        span, start_time = self._spans.pop(span_id)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        if self.enabled and span:
            span.end(
                output=output_data,
                metadata={"latency_ms": latency_ms, "status": status}
            )
        else:
            print(f"  [SPAN END]   id={span_id[:8]}... latency={latency_ms}ms status={status}")

    # ─── LLM GENERATION LOGGING ──────────────────────────────

    def log_llm_call(self, trace_id: str, node_name: str, model: str,
                     prompt: str, response: str, tokens_used: int = 0):
        """
        Log a single LLM generation call as a generation event.
        Enables token cost tracking and quality scoring in Langfuse.
        """
        if self.enabled and self._current_trace:
            self._current_trace.generation(
                name=f"{node_name}_llm_call",
                model=model,
                model_parameters={"temperature": 0},
                input=prompt,
                output=response,
                usage={
                    "total_tokens": tokens_used,
                    "unit": "TOKENS"
                }
            )
        else:
            print(f"  [LLM CALL]   node='{node_name}' model='{model}' tokens={tokens_used}")
            print(f"               prompt_preview='{prompt[:60]}...' " if len(prompt) > 60 else f"               prompt='{prompt}'")
            print(f"               response_preview='{response[:60]}...' " if len(response) > 60 else f"               response='{response}'")

    # ─── SCORE / FEEDBACK LOGGING ────────────────────────────

    def log_score(self, trace_id: str, name: str, value: float, comment: str = ""):
        """
        Attach a numeric quality score to a trace.
        Used by Self-RAG graders to record relevance/hallucination metrics.
        """
        if self.enabled and self._current_trace:
            self.client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment
            )
        else:
            print(f"  [SCORE]      trace={trace_id[:8]}... name='{name}' value={value} comment='{comment}'")

    def flush(self):
        """Force-flush all pending events to Langfuse cloud."""
        if self.enabled and self.client:
            self.client.flush()


# ─── MODULE-LEVEL SINGLETON ──────────────────────────────────
# Import this instance across all OmniBrain modules
langfuse = LangfuseClient()


# ─── DECORATOR: Trace Any Node Function ──────────────────────

def traced_node(node_name: str):
    """
    Decorator to automatically wrap a LangGraph node function
    with Langfuse start/end span tracing.

    Usage:
        @traced_node("search_agent")
        def search_agent_node(state):
            ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(state: Dict[str, Any]):
            trace_id = state.get("trace_id", "no-trace")
            span_id = langfuse.start_span(
                trace_id=trace_id,
                name=node_name,
                input_data={"question": state.get("question", ""), "route": state.get("route", "")}
            )
            try:
                result = fn(state)
                langfuse.end_span(span_id, output_data=result or {}, status="success")
                return result
            except Exception as e:
                langfuse.end_span(span_id, output_data={"error": str(e)}, status="error")
                raise
        return wrapper
    return decorator


# ─── VERIFICATION DEMO ───────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("   OmniBrain — Langfuse Integration Verification")
    print("   Member 1: Vasu Sree")
    print("=" * 60)

    client = LangfuseClient()

    # Simulate a full trace lifecycle
    trace_id = client.start_trace(
        name="omnibrain_query",
        input_data={"question": "Explain reinforcement learning from the document"},
        session_id="test-session-001"
    )

    # Simulate Supervisor span
    sup_span = client.start_span(trace_id, "supervisor_node", {"question": "Explain reinforcement learning"})
    time.sleep(0.05)
    client.end_span(sup_span, {"route": "search"})

    # Simulate Search Agent span
    search_span = client.start_span(trace_id, "search_agent_node", {"question": "Explain reinforcement learning"})
    time.sleep(0.08)
    client.log_llm_call(
        trace_id=trace_id,
        node_name="search_agent",
        model="gpt-4o-mini",
        prompt="Extract answer from document: Explain reinforcement learning",
        response="Reinforcement learning is a subfield of machine learning...",
        tokens_used=142
    )
    client.end_span(search_span, {"documents_found": 2})

    # Simulate Self-RAG grading score
    client.log_score(trace_id, "document_relevance", 1.0, "Both documents relevant")
    client.log_score(trace_id, "hallucination_check", 1.0, "No hallucination detected")
    client.log_score(trace_id, "answer_utility", 1.0, "Answer fully addresses query")

    # End trace
    client.end_trace(trace_id, {"response": "Reinforcement learning is a..."}, status="success")

    client.flush()
    print("\n[DONE] Langfuse trace simulation complete.")
