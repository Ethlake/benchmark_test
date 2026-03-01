"""Utility tools — agent lifecycle, not pipeline-phase-specific."""

from dataagent_benchmark.tools.utilities.rollback import rollback
from dataagent_benchmark.tools.utilities.think import think


def _register_utility_tools():
    """Return (fn, description) pairs for utility tools."""
    return [
        (
            think,
            "Think step-by-step about your next action. Use this before choosing a tool to reason about the current state, what the data tells you, and what to do next.",
        ),
        (
            rollback,
            "Undo the last N transforms on a dataset, restoring a prior version without losing all work.",
        ),
    ]
