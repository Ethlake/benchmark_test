"""submit_finetune tool — run a multimodal finetune job.

Delegates execution to the ``curation-train`` CLI via ``uv run --project``
so the train package's own venv (with torch, deepspeed, etc.) is used.
The agent package has zero heavy ML dependencies.

Blocks until the job finishes and returns the final result.
"""

import json
import subprocess
import uuid
from pathlib import Path
from typing import Annotated, Any
import os
import structlog

from dataagent_benchmark.domain.artifacts import ArtifactRef, StepResult
from dataagent_benchmark.domain.tool_context import FromEnv, PerDataset
from dataagent_benchmark.infra.gym_env import _find_repo_root

log = structlog.get_logger()


def submit_finetune(
    dataset_id: Annotated[str, "Dataset name from the working store"],
    output_dir: Annotated[str, "Output directory (auto-generated if empty)"] = "",
    # --- Injected by env ---
    dataset: Annotated[Any, PerDataset("working")] = None,
    training_config: Annotated[dict, FromEnv("training_config")] = None,
    run_dir: Annotated[Any, FromEnv("run_dir")] = None,
) -> StepResult | str:
    """Run a multimodal finetune job (blocking).

    Supports ``method=qwen`` and ``method=llava`` via ``training_config``.
    Uses fixed hyperparameters by default; limited smoke-test overrides
    (e.g. ``max_steps``) are passed through when provided.
    """
    training_cfg = training_config or {}
    gpu_id = int(training_cfg.get("gpu_id", 0))
    method = str(training_cfg.get("method", "qwen")).lower().strip()
    if method in {"qwen2.5", "qwen25"}:
        method = "qwen"
    if method in {"llava1.5", "llava-1.5"}:
        method = "llava"

    default_model = (
        "llava-hf/llava-1.5-7b-hf" if method == "llava" else "Qwen/Qwen2.5-VL-7B-Instruct"
    )
    default_ds_cfg = "zero2" if method == "llava" else "zero3"

    job_id = f"ft-{uuid.uuid4().hex[:8]}"
    if output_dir:
        output = output_dir
    elif run_dir is not None:
        finetune_dir = run_dir / "finetune"
        finetune_dir.mkdir(parents=True, exist_ok=True)
        output = str(finetune_dir / job_id)
    else:
        output = f"./finetune-{dataset_id}-{job_id}"

    # Add model-family suffix for easier run browsing
    output_lc = output.lower()
    if method == "llava":
        if "llava" not in output_lc:
            output = output.rstrip("/") + "-llava15"
    else:
        if "qwen25" not in output_lc and "qwen2.5" not in output_lc:
            output = output.rstrip("/") + "-qwen25vl"

    ds = dataset

    # Save Arrow dataset to disk for the training script
    data_path = str(Path(output) / "data")
    Path(data_path).mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(data_path)

    # Build config JSON from session training_config + defaults
    config = {
        "method": method,
        "data_path": data_path,
        "output_dir": output,
        "model_name_or_path": training_cfg.get("model_name_or_path", default_model),
        "deepspeed_config": training_cfg.get("deepspeed_config", default_ds_cfg),
    }
    for key in ("max_steps", "save_steps", "logging_steps"):
        if key in training_cfg:
            config[key] = training_cfg[key]

    # Write config JSON for the CLI
    work_dir = Path(output)
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / "finetune_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False))

    log.info(
        "finetune_start",
        job_id=job_id,
        method=method,
        model=config["model_name_or_path"],
        dataset=dataset_id,
        rows=len(ds),
    )

    train_project = str(_find_repo_root() / "packages" / "train")

    try:
        gpu_spec = training_cfg.get("gpu_ids", training_cfg.get("gpu_id", 0))

        if isinstance(gpu_spec, list):
            cuda_visible = ",".join(str(x) for x in gpu_spec)
        else:
            # 允许 gpu_id: 3 或 gpu_id: "0,1,2,3"
            cuda_visible = str(gpu_spec).strip()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible

        proc = subprocess.run(
            ["uv", "run", "--project", train_project, "curation-train", str(config_path)],
            capture_output=True,
            text=True,
            timeout=7200,
            cwd=work_dir,
            env=env,
        )

        if proc.returncode != 0:
            error_output = (proc.stderr or proc.stdout or "")[-2000:]
            return json.dumps(
                {
                    "job_id": job_id,
                    "status": "failed",
                    "error": f"curation-train exited {proc.returncode}: {error_output}",
                }
            )

        # Parse the JSON result from stdout (last line)
        output_lines = proc.stdout.strip().splitlines()
        parsed = json.loads(output_lines[-1])

        if parsed.get("status") == "failed":
            error_msg = parsed.get("error", "unknown error")
            # Truncate long tracebacks for the LLM
            if len(error_msg) > 500:
                error_msg = error_msg[:500] + "..."
            return json.dumps(
                {
                    "job_id": job_id,
                    "status": "failed",
                    "error": error_msg,
                }
            )

        adapter_path = parsed.get("metrics", {}).get("output_dir", output)

        payload = json.dumps(
            {
                "job_id": job_id,
                "method": method,
                "status": parsed["status"],
                "metrics": parsed.get("metrics", {}),
                "adapter_path": adapter_path,
            }
        )
        return StepResult(
            payload=payload,
            created=[ArtifactRef("model", job_id, 0)],
            updates={"active_model": ArtifactRef("model", job_id, 0)},
            metrics={"job_id": job_id, "method": method, "adapter_path": adapter_path},
        )

    except Exception as exc:
        log.exception("finetune_failed", job_id=job_id)
        return json.dumps(
            {
                "job_id": job_id,
                "status": "failed",
                "error": str(exc),
            }
        )
