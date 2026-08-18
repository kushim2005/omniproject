# ============================================================
# OmniBrain — Week 3
# Member 3: Chaitanya
# Task: Self-Correction Loop Integration
# ============================================================

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os

class SelfCorrector:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.use_llm = self.api_key is not None
        
        if self.use_llm:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, openai_api_key=self.api_key)
            
            # Prompt instructions for self-correction of responses
            correction_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an AI assistant specialized in self-correction for RAG systems.\n"
                           "Your task is to analyze a generated answer against a set of retrieved facts and the original question.\n"
                           "Identify and rewrite any parts of the answer that are not supported by the facts (hallucinations) "
                           "or do not address the question. Ensure the output is strictly grounded in the facts provided. "
                           "Output ONLY the corrected answer."),
                ("user", "User Question: {question}\n\nRetrieved Facts:\n{facts}\n\nFailed Answer:\n{answer}\n\nCorrected Answer:")
            ])
            self.correction_chain = correction_prompt | self.llm
        else:
            print("[WARN] No API key detected. Self-Corrector running in heuristic mode.")

    def correct_answer(self, question: str, facts: list, answer: str) -> str:
        """
        Corrects a generated answer to ensure it matches the facts.
        """
        facts_str = "\n".join(facts)
        if self.use_llm:
            try:
                res = self.correction_chain.invoke({
                    "question": question,
                    "facts": facts_str,
                    "answer": answer
                })
                return res.content.strip()
            except Exception as e:
                print(f"[ERROR] LLM Self-Correction failed: {e}")
                
        # Heuristic fallback: construct answer directly from facts
        if facts:
            return f"Based strictly on document records: {' '.join(facts)}"
        return "No relevant context was found to safely answer the question without hallucination."
