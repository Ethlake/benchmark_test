"""Postselection tools — saving, formatting, finetune, eval."""

from dataagent_benchmark.tools.postselection.convert_format import convert_format
from dataagent_benchmark.tools.postselection.submit_eval import submit_eval
from dataagent_benchmark.tools.postselection.submit_finetune import submit_finetune
from dataagent_benchmark.tools.preselection.save_to_disk import save_to_disk
from dataagent_benchmark.tools.preselection.save_to_hf import save_to_hf


def _register_postselection_tools():
    """Return (fn, description) pairs for postselection tools."""
    return [
        (
            save_to_disk,
            "Concatenate working copies of selected datasets and save locally as parquet or JSONL.",
        ),
        (
            save_to_hf,
            "Concatenate working copies of selected datasets and push to HuggingFace Hub.",
        ),
        (
            convert_format,
            "Convert dataset rows into a unified chat instruction format (chatml/llava/qwen).",
        ),
        (
            submit_finetune,
            "Run a multimodal finetune job (blocking). Uses fixed hyperparameters. Returns final status and metrics.",
        ),
        (
            submit_eval,
            "Run a VLMEvalKit evaluation job for a finetuned model (blocking). Returns final benchmark results.",
        ),
    ]
