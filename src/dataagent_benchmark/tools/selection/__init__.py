"""Selection tools — quality diagnosis, data cleaning, and sampling."""

from dataagent_benchmark.tools.selection.balanced_sampling import balanced_sampling
from dataagent_benchmark.tools.selection.dedup_check import dedup_check
from dataagent_benchmark.tools.selection.explore_dataset import explore_dataset
from dataagent_benchmark.tools.selection.integrity_check import integrity_check
from dataagent_benchmark.tools.selection.text_quality_check import text_quality_check
from dataagent_benchmark.tools.selection.transform_dataset import transform_dataset


def _register_selection_tools():
    """Return (fn, description) pairs for selection tools."""
    return [
        (
            integrity_check,
            "Check structural integrity of a dataset: required fields, answer markers.",
        ),
        (
            text_quality_check,
            "Measure text quality: answer lengths, reasoning steps, language heuristics.",
        ),
        (dedup_check, "Measure exact and near-duplicate rates in a dataset."),
        (explore_dataset, "Compute column-level statistics from the dataset."),
        (transform_dataset, "Apply a row-level transformation to a dataset's working copy."),
        (
            balanced_sampling,
            "Rebalance sample allocations so dominant subsets don't overwhelm training.",
        ),
    ]
