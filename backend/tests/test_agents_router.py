import asyncio
import unittest

from app.agents.router import RuleBasedRouter
from app.agents.state import AgentType


class RuleBasedRouterTests(unittest.TestCase):
    def test_routes_multiple_agents_in_priority_order(self):
        router = RuleBasedRouter()
        decision = asyncio.run(
            router.route("Please summarize this chart and find the relevant page")
        )

        self.assertEqual(
            [agent for agent in decision.selected_agents],
            [AgentType.SUMMARY, AgentType.VISION, AgentType.SEARCH],
        )


if __name__ == "__main__":
    unittest.main()
