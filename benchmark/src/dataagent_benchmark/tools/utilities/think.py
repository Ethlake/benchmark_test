"""think — explicit reasoning tool for models without native thought output.

Forces the LLM to articulate its reasoning before choosing the next tool.
The reasoning is logged as a ``thought`` structlog event and returned as-is
so the LLM can reference it in subsequent turns.
"""

from typing import Annotated

import structlog

log = structlog.get_logger()


def think(
    reasoning: Annotated[
        str,
        "Your step-by-step reasoning about what to do next. "
        "Analyze the current state, what you've learned so far, "
        "and explain which tool you plan to call next and why.",
    ],
) -> str:
    """Think step-by-step about your next action.

    Use this tool to reason about the current state of the curation task
    before choosing your next tool. Explain what you've observed, what
    the data tells you, and what action you plan to take next.
    """
    log.info("thought", text=reasoning)
    return reasoning
