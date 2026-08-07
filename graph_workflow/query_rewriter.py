# ============================================================
# OmniBrain — Week 3
# Member 2: Vasu Sree
# Task: Query Rewrite Agent
# ============================================================

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import re

class QueryRewriter:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.use_llm = self.api_key is not None
        
        if self.use_llm:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=self.api_key)
            
            # Prompt instructions for rewriting queries
            rewrite_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an AI assistant specialized in rewriting search queries for RAG vector stores.\n"
                           "Your task is to analyze the user's current question and rewrite it to make it highly optimized "
                           "for semantic vector retrieval. Strip out conversational filler words, question words (who, what, where), "
                           "and keep only the critical nouns, verbs, and keywords. Output ONLY the rewritten query text."),
                ("user", "Original Query: {query}\nRewritten Query:")
            ])
            self.rewrite_chain = rewrite_prompt | self.llm
        else:
            print("[WARN] No API key detected. Query Rewriter running in heuristic mode.")

    def rewrite_query(self, query: str) -> str:
        """
        Rewrites a search query to optimize vector database retrieval.
        """
        if self.use_llm:
            try:
                res = self.rewrite_chain.invoke({"query": query})
                return res.content.strip()
            except Exception as e:
                print(f"[ERROR] LLM Query Rewriting failed: {e}")
                
        # Heuristic fallback rules (offline mode)
        clean = query.lower().strip()
        
        # Strip common conversation/question starters
        starters = [
            r"^where is the\s+", r"^what is the\s+", r"^show me the\s+", 
            r"^can you find\s+", r"^find information about\s+", r"^tell me about\s+",
            r"^search for\s+", r"^give me the\s+", r"^retrieve\s+"
        ]
        for pattern in starters:
            clean = re.sub(pattern, "", clean)
            
        # Clean special chars and return
        clean = re.sub(r'[^\w\s]', '', clean)
        return clean.strip() or query
