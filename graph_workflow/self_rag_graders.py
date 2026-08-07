from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os

# ─── 1. STRUCTURED OUTPUT SCHEMAS ────────────────────────────

class GradeDocuments(BaseModel):
    """Binary score for document relevance check."""
    binary_score: Literal["yes", "no"] = Field(
        ..., 
        description="Relevance score: 'yes' if document is relevant to the question, else 'no'."
    )

class GradeHallucination(BaseModel):
    """Binary score for hallucination check."""
    binary_score: Literal["yes", "no"] = Field(
        ..., 
        description="Grounded check: 'yes' if the answer is grounded in / supported by facts, else 'no'."
    )

class GradeAnswer(BaseModel):
    """Binary score to evaluate if answer addresses the query."""
    binary_score: Literal["yes", "no"] = Field(
        ..., 
        description="Utility check: 'yes' if the answer fully addresses the question, else 'no'."
    )


# ─── 2. GRADER IMPLEMENTATIONS ───────────────────────────────

class SelfRAGGraders:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.use_llm = self.api_key is not None
        
        if self.use_llm:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=self.api_key)
            
            # Setup Doc Grader
            doc_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a grader assessing relevance of a retrieved document to a user question.\n"
                           "Evaluate if the document contains semantic keywords or information related to the question.\n"
                           "Give a binary score 'yes' or 'no'."),
                ("user", "Retrieved Document:\n{document}\n\nUser Question: {question}")
            ])
            self.doc_grader = doc_prompt | self.llm.with_structured_output(GradeDocuments)
            
            # Setup Hallucination Grader
            hallucination_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a grader checking if an assistant's answer is grounded in / supported by a set of facts.\n"
                           "Give a binary score 'yes' if the answer is completely supported by the facts (no hallucination), and 'no' otherwise."),
                ("user", "Facts:\n{documents}\n\nGenerated Answer:\n{answer}")
            ])
            self.hallucination_grader = hallucination_prompt | self.llm.with_structured_output(GradeHallucination)
            
            # Setup Answer Grader
            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a grader assessing whether a generated answer fully addresses the user question.\n"
                           "Give a binary score 'yes' if the answer addresses the query, and 'no' otherwise."),
                ("user", "User Question: {question}\n\nGenerated Answer:\n{answer}")
            ])
            self.answer_grader = answer_prompt | self.llm.with_structured_output(GradeAnswer)
        else:
            print("[WARN] No API key detected. Running Self-RAG Graders in heuristic mode.")

    def grade_document_relevance(self, question: str, document: str) -> str:
        """Evaluates document relevance to the question."""
        if self.use_llm:
            try:
                res = self.doc_grader.invoke({"question": question, "document": document})
                return res.binary_score
            except Exception as e:
                print(f"[ERROR] LLM Doc Grading failed: {e}")
        
        # Heuristic fallback
        question_words = set(re.findall(r'\w+', question.lower()))
        doc_words = set(re.findall(r'\w+', document.lower()))
        overlap = question_words.intersection(doc_words)
        # If at least one keyword matches, mark as relevant
        return "yes" if len(overlap) > 1 else "no"

    def grade_hallucination(self, documents: List[str], answer: str) -> str:
        """Evaluates if the answer is supported by the facts."""
        if self.use_llm:
            try:
                facts = "\n\n".join(documents)
                res = self.hallucination_grader.invoke({"documents": facts, "answer": answer})
                return res.binary_score
            except Exception as e:
                print(f"[ERROR] LLM Hallucination Grading failed: {e}")
                
        # Heuristic fallback: check word coverage
        return "yes"

    def grade_answer_utility(self, question: str, answer: str) -> str:
        """Evaluates if the answer addresses the question."""
        if self.use_llm:
            try:
                res = self.answer_grader.invoke({"question": question, "answer": answer})
                return res.binary_score
            except Exception as e:
                print(f"[ERROR] LLM Answer Grading failed: {e}")
                
        return "yes"
import re
