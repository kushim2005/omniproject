"""
Rule‑based router for the Supervisor Agent.
Implements an abstract RouterInterface to allow future replacement
with an LLM‑based or LangGraph router.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import List

from app.agents.state import AgentType, RoutingDecision

logger = logging.getLogger(__name__)


class RouterInterface(ABC):
    """Abstract interface for any routing strategy (rule‑based, ML, LLM, etc.)."""

    @abstractmethod
    async def route(self, query: str) -> RoutingDecision:
        """
        Determine which agents should handle the given user query.

        Args:
            query: The user's input text.

        Returns:
            RoutingDecision containing selected agents, confidence, and reason.
        """
        pass


class RuleBasedRouter(RouterInterface):
    """
    A keyword‑based router that classifies the query and returns a RoutingDecision.
    Uses configurable keyword lists for each agent type.
    """

    def __init__(self):
        # Keyword mappings – each agent type has a set of trigger words/phrases.
        self.keyword_map = {
            AgentType.VISION: {
                "figure",
                "chart",
                "image",
                "diagram",
                "graph",
                "plot",
                "illustration",
                "picture",
                "photo",
            },
            AgentType.SUMMARY: {
                "summary",
                "summarize",
                "overview",
                "recap",
                "synopsis",
                "abstract",
                "brief",
            },
            AgentType.SEARCH: {
                "page",
                "section",
                "paragraph",
                "what does",
                "where",
                "who",
                "when",
                "why",
                "how",
                "find",
                "retrieve",
                "look up",
                "search",
            },
            AgentType.SQL: {
                "table",
                "sql",
                "database",
                "rows",
                "columns",
                "query",
                "tabular",
                "dataframe",
                "spreadsheet",
            },
        }
        # Compile regex patterns for each keyword for faster matching (optional)
        self.patterns = {
            agent: re.compile(r"|".join(r"\b" + re.escape(kw) + r"\b" for kw in keywords), re.IGNORECASE)
            for agent, keywords in self.keyword_map.items()
        }

    async def route(self, query: str) -> RoutingDecision:
        """
        Analyze the query, detect keywords, and return a routing decision.

        The decision includes all matching agent types. If no match, returns UNKNOWN.
        Confidence is computed based on the number of distinct agent matches.
        """
        logger.debug(f"Routing query: {query[:100]}...")

        matched_agents: List[AgentType] = []
        reasons: List[str] = []
        priority_order = [
            AgentType.SUMMARY,
            AgentType.VISION,
            AgentType.SEARCH,
            AgentType.SQL,
        ]

        for agent in priority_order:
            pattern = self.patterns[agent]
            if pattern.search(query):
                matched_agents.append(agent)
                reasons.append(f"Matched keywords for {agent.value}")

        if not matched_agents:
            matched_agents = [AgentType.UNKNOWN]
            reason = "No relevant keywords detected; routing to UNKNOWN."
            confidence = 0.3
        else:
            if len(matched_agents) == 1:
                confidence = 0.95
            else:
                confidence = 0.80 + (0.05 * (len(matched_agents) - 1))
                confidence = min(confidence, 0.99)
            reason = "; ".join(reasons)

        decision = RoutingDecision(
            selected_agents=matched_agents,
            confidence=round(confidence, 2),
            reason=reason,
        )

        logger.info(f"Routing decision: {decision.selected_agents} (confidence={decision.confidence})")
        return decision