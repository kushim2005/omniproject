"""
Supervisor Agent responsible for orchestrating the routing of user queries
to the appropriate downstream agents.

This class depends on an abstract RouterInterface, allowing the routing
logic to be replaced (e.g., with an LLM‑based router) in future weeks.
"""

import logging
import time
from uuid import uuid4
from typing import Optional
from uuid import UUID

from app.agents.state import AgentState, RoutingDecision
from app.agents.router import RouterInterface

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """
    The Supervisor Agent decides which agents should handle a given user query.
    It maintains state and logs execution timing.
    """

    def __init__(self, router: RouterInterface):
        """
        Initialize the Supervisor with a routing strategy.

        Args:
            router: An implementation of RouterInterface (e.g., RuleBasedRouter).
        """
        self.router = router
        logger.info("SupervisorAgent initialized with router: %s", router.__class__.__name__)

    async def process_query(
        self,
        query: str,
        conversation_id: Optional[UUID] = None,
        document_id: Optional[UUID] = None,
    ) -> RoutingDecision:
        """
        Process a user query: create state, run the router, log, and return the decision.

        Args:
            query: The user's input text.
            conversation_id: Optional; if not provided, a new UUID is generated.
            document_id: Optional target document ID.

        Returns:
            RoutingDecision containing selected agents, confidence, and reason.

        Note:
            This method does NOT retrieve documents, generate answers, or call any AI.
            It only determines the routing.
        """
        start_time = time.perf_counter()

        # Generate a conversation ID if not provided
        conv_id = conversation_id or uuid4()

        # Initialize state (future steps will enrich this state)
        state = AgentState(
            conversation_id=conv_id,
            document_id=document_id,
            user_query=query,
        )
        logger.debug(f"Created AgentState for conversation {conv_id}")

        # Perform routing
        decision = await self.router.route(query)

        # Update state with routing results
        state.selected_agents = decision.selected_agents

        # Record processing time
        elapsed = time.perf_counter() - start_time
        state.processing_time = elapsed

        logger.info(
            f"Supervisor processed query in {elapsed:.4f}s, "
            f"routed to {[a.value for a in decision.selected_agents]}"
        )

        return decision

    # For future extension: method to update state with results from downstream agents
    # async def update_state(self, state: AgentState, agent_output: Any) -> AgentState:
    #     ...