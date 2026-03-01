"""Preselection tools — profiling, mixing, filtering, and VLM ops."""

from dataagent_benchmark.tools.preselection.compute_mix_ratio import compute_mix_ratio
from dataagent_benchmark.tools.preselection.inspect_samples import inspect_samples
from dataagent_benchmark.tools.preselection.load_dataset import load_dataset
from dataagent_benchmark.tools.preselection.profile_datasets import profile_datasets
from dataagent_benchmark.tools.preselection.quality_filter import quality_filter
from dataagent_benchmark.tools.preselection.vlm_filter import vlm_filter
from dataagent_benchmark.tools.preselection.vlm_mix import vlm_mix
from dataagent_benchmark.tools.preselection.vlm_modify import vlm_modify


def _register_preselection_tools():
    """Return (fn, description) pairs for preselection tools."""
    return [
        (load_dataset, "Load a single dataset into the working store."),
        (
            profile_datasets,
            "Load and profile all candidate datasets. Returns row counts, columns, and quality scores derived from existing metadata.",
        ),
        (inspect_samples, "View example rows from a dataset with automated quality notes."),
        (compute_mix_ratio, "Compute a data mixture recipe with per-dataset sample allocations."),
        (
            vlm_filter,
            "Filter a dataset by image integrity checks. Removes missing/corrupt images, images with sides below 200px, and extreme aspect ratios (>3:1).",
        ),
        (
            quality_filter,
            "Filter a dataset by text quality, alignment, and deduplication. Removes N/A answers, spam, short text, and near-duplicate images and text.",
        ),
        (vlm_mix, "Sample K rows from each dataset and concatenate into one combined dataset."),
        (
            vlm_modify,
            "Rewrite or augment text annotations using a frontier vision model. Sends images + prompt to a VLM and replaces the text column with the response.",
        ),
    ]
