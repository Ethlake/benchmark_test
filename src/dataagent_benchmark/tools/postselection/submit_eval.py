"""submit_eval tool — run a VLMEvalKit evaluation job.

Runs VLMEvalKit directly from the repo submodule at
``packages/VLMEvalKit`` using the configured Conda Python environment.

Supports both Qwen and LLaVA model families.
"""

import json
import os
import subprocess
import textwrap
import uuid
from pathlib import Path
from typing import Annotated, Any

import structlog

from dataagent_benchmark.domain.artifacts import ArtifactRef, StepResult
from dataagent_benchmark.domain.tool_context import FromArtifact, FromEnv
from dataagent_benchmark.infra.gym_env import _find_repo_root

log = structlog.get_logger()

_DEFAULT_BENCHMARKS = [
    "MMStar",
    "OCRBench",
    "MathVista_MINI",
    "HallusionBench",
    "MMVet",
    "MMMU_DEV_VAL",
]

_TIMEOUT_SECONDS = 14400


def _resolve_vlmeval_dir(eval_config: dict, repo_root: Path) -> Path:
    """Resolve VLMEvalKit directory, preferring the in-repo submodule."""
    submodule = repo_root / "packages" / "VLMEvalKit"
    if submodule.is_dir():
        return submodule

    configured = eval_config.get("vlmeval_dir") or os.environ.get("VLMEVAL_DIR")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_dir():
            return candidate
        raise RuntimeError(f"VLMEvalKit directory not found: {candidate}")

    raise RuntimeError(
        "VLMEvalKit directory not configured. Add submodule at "
        "'packages/VLMEvalKit' or set eval.vlmeval_dir / VLMEVAL_DIR."
    )


def _resolve_vlmeval_python(eval_config: dict) -> str:
    """Resolve Python path with VLMEvalKit deps, preferring the Conda env."""
    configured = eval_config.get("vlmeval_python") or os.environ.get("VLMEVAL_PYTHON")
    if configured:
        return configured

    default_conda_python = Path("/home/adanato/miniconda3/envs/vlmeval/bin/python")
    if default_conda_python.exists():
        return str(default_conda_python)

    raise RuntimeError(
        "VLMEVAL_PYTHON not configured and default Conda env python not found at "
        "/home/adanato/miniconda3/envs/vlmeval/bin/python."
    )


def _resolve_model_family(model_path: str, requested: str) -> str:
    raw = (requested or "auto").strip().lower()
    if raw in {"qwen", "llava"}:
        return raw
    if raw != "auto":
        raise RuntimeError(
            f"Unsupported model_family '{requested}'. Use 'auto', 'qwen', or 'llava'."
        )
    lowered = model_path.lower()
    if "llava" in lowered:
        return "llava"
    return "qwen"


def _write_driver_script(
    *,
    script_path: Path,
    vlmeval_dir: Path,
    model_path: str,
    model_name: str,
    benchmarks: list[str],
    output_dir: Path,
    max_samples: int,
    mode: str,
    model_family: str,
    use_vllm: bool,
    api_nproc: int = 16,
) -> None:
    """Write a small VLMEvalKit driver script with dynamic model registration."""
    if model_family == "qwen":
        registration = textwrap.dedent(
            f"""\
            from vlmeval.vlm.qwen2_vl import Qwen2VLChat
            supported_VLM[{model_name!r}] = partial(
                Qwen2VLChat,
                model_path={model_path!r},
                use_vllm={use_vllm!r},
            )
            """
        )
    else:
        # LLaVA 1.5 uses LlavaForConditionalGeneration + AutoProcessor,
        # but VLMEvalKit's LLaVA_Next dispatches on string matching in
        # model_path and defaults to LlavaNext* classes. We inline a thin
        # wrapper that uses the correct HF classes for llava-1.5.
        registration = textwrap.dedent(
            f"""\
            import torch, warnings
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            from vlmeval.vlm.base import BaseModel

            class _LLaVA15Eval(BaseModel):
                INSTALL_REQ = False
                INTERLEAVE = True

                def __init__(self, model_path, **kwargs):
                    self.model_path = model_path
                    self.processor = AutoProcessor.from_pretrained(model_path)
                    try:
                        import flash_attn
                        attn = {{"attn_implementation": "flash_attention_2"}}
                    except ImportError:
                        attn = {{}}
                    model = LlavaForConditionalGeneration.from_pretrained(
                        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **attn,
                    )
                    self.model = model.eval().cuda()
                    self.kwargs = dict(do_sample=False, temperature=0, max_new_tokens=2048,
                                       top_p=None, num_beams=1, **kwargs)

                def generate_inner(self, message, dataset=None):
                    from PIL import Image
                    images, prompt_parts = [], []
                    for item in message:
                        if item["type"] == "image":
                            images.append(Image.open(item["value"]).convert("RGB"))
                        elif item["type"] == "text":
                            prompt_parts.append(item["value"])
                    prompt = "\\n".join(prompt_parts)
                    conversation = [
                        {{"role": "user", "content": [
                            *[{{"type": "image"}} for _ in images],
                            {{"type": "text", "text": prompt}},
                        ]}},
                    ]
                    text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                    inputs = self.processor(text=text, images=images or None, return_tensors="pt").to(self.model.device)
                    with torch.no_grad():
                        output = self.model.generate(**inputs, **self.kwargs)
                    input_len = inputs["input_ids"].shape[1]
                    return self.processor.decode(output[0][input_len:], skip_special_tokens=True)

            supported_VLM[{model_name!r}] = partial(_LLaVA15Eval, model_path={model_path!r})
            """
        )

    script = "\n".join(
        [
            "import importlib.util",
            "import os",
            "import sys",
            "",
            f"vlmeval_dir = {str(vlmeval_dir)!r}",
            "sys.path.insert(0, vlmeval_dir)",
            "",
            "from functools import partial",
            "from vlmeval.config import supported_VLM",
            registration.rstrip(),
            "",
            f"benchmarks = {benchmarks!r}",
            "sys.argv = ['run.py']",
            "sys.argv += ['--data'] + benchmarks",
            f"sys.argv += ['--model', {model_name!r}]",
            f"sys.argv += ['--work-dir', {str(output_dir)!r}]",
            f"sys.argv += ['--mode', {mode!r}]",
            f"sys.argv += ['--api-nproc', '{api_nproc}']",
            # NOTE: Do NOT pass --use-vllm via sys.argv. VLMEvalKit leaks it
            # into judge_kwargs, breaking the OpenAI GPT-4 judge API call.
            # Instead, use_vllm is set directly on the model via partial().
            f"if {use_vllm!r}:",
            "    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')",
            "",
            "from vlmeval.smp import load_env",
            "",
            "os.chdir(vlmeval_dir)",
            "load_env()",
            "",
            "spec = importlib.util.spec_from_file_location('vlmeval_run', os.path.join(vlmeval_dir, 'run.py'))",
            "run_module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(run_module)",
            "",
            f"max_samples = {max_samples!r}",
            "if max_samples > 0:",
            "    _orig_build_dataset = run_module.build_dataset",
            "",
            "    def _build_dataset_10(name, **kwargs):",
            "        ds = _orig_build_dataset(name, **kwargs)",
            "        if hasattr(ds, 'data') and len(ds.data) > max_samples:",
            "            ds.data = ds.data.iloc[:max_samples].reset_index(drop=True)",
            "        return ds",
            "",
            "    run_module.build_dataset = _build_dataset_10",
            "",
            "run_module.main()",
            "",
        ]
    )
    script_path.write_text(script, encoding="utf-8")


def _parse_score_from_file(result_file: Path) -> float | None:
    """Parse one result file produced by VLMEvalKit."""
    if result_file.suffix == ".json":
        data = json.loads(result_file.read_text(encoding="utf-8"))
        score = data.get("Final Score")
        if score is None:
            return None
        try:
            return float(score)
        except (TypeError, ValueError):
            return None

    if result_file.suffix != ".csv":
        return None

    import csv

    with result_file.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    # *_acc.csv (MCQ benchmarks)
    if "Overall" in rows[0]:
        try:
            return float(rows[0]["Overall"])
        except (TypeError, ValueError):
            pass

    # *_score.csv with acc/aAcc on "Overall" row
    for key in ("acc", "aAcc"):
        if key not in rows[0]:
            continue
        for row in rows:
            marker = row.get("split") or row.get("Category") or row.get("Task&Skill")
            if marker == "Overall":
                try:
                    return float(row[key])
                except (TypeError, ValueError):
                    break
        try:
            return float(rows[0][key])
        except (TypeError, ValueError):
            pass

    return None


def _find_latest_result_file(output_dir: Path, model_name: str, benchmark: str) -> Path | None:
    prefix = f"{model_name}_{benchmark}"
    patterns = [
        f"{prefix}_score.json",
        f"{prefix}_acc.csv",
        f"{prefix}*_score.csv",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(output_dir.rglob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _collect_metrics(output_dir: Path, model_name: str, benchmarks: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for benchmark in benchmarks:
        result_file = _find_latest_result_file(output_dir, model_name, benchmark)
        if result_file is None:
            continue
        score = _parse_score_from_file(result_file)
        if score is not None:
            metrics[benchmark] = round(score, 4)
    return metrics


def _list_result_files(output_dir: Path, limit: int = 50) -> list[str]:
    if not output_dir.exists():
        return []
    files = [p for p in output_dir.rglob("*") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p.relative_to(output_dir)) for p in files[:limit]]


def submit_eval(
    model_id: Annotated[str, "Model artifact ID — the job_id returned by submit_finetune (e.g. 'ft-abc123')"],
    model_family: Annotated[str, "Model family: auto (default), qwen, or llava"] = "auto",
    # --- Injected by env ---
    adapter_path: Annotated[str, FromArtifact("model", "adapter_path", "model_id")] = "",
    eval_config: Annotated[dict, FromEnv("eval_config")] = None,
    run_dir: Annotated[Any, FromEnv("run_dir")] = None,
) -> StepResult | str:
    """Run a VLMEvalKit evaluation job for a finetuned model (blocking).

    Runs configured benchmarks and returns parsed metric outputs.
    """
    repo_root = _find_repo_root()

    # Read eval config
    eval_cfg = eval_config or {}
    benchmarks = eval_cfg.get("benchmarks", _DEFAULT_BENCHMARKS)
    mode = str(eval_cfg.get("mode", "all"))
    use_vllm = bool(eval_cfg.get("use_vllm", True))
    max_samples = int(eval_cfg.get("max_samples", 0) or 0)
    gpu_id = int(eval_cfg.get("gpu_id", 0))
    api_nproc = int(eval_cfg.get("api_nproc", 16))
    family_cfg = str(eval_cfg.get("model_family", model_family))
    resolved_family = _resolve_model_family(adapter_path, family_cfg)
    # Honor eval.use_vllm for all model families; backend compatibility is
    # handled by VLMEvalKit / model implementation at runtime.
    effective_use_vllm = use_vllm

    job_id = f"eval-{uuid.uuid4().hex[:8]}"

    # Prepare output directory
    if run_dir is not None:
        eval_dir = run_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        work_dir = eval_dir / job_id
    else:
        work_dir = Path(f"./eval-{job_id}")
    work_dir.mkdir(parents=True, exist_ok=True)
    work_dir = work_dir.resolve()

    model_name = f"custom_{resolved_family}_{job_id}"
    results_dir = work_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Keep run config on disk for debugging/repro
    run_config = {
        "model_path": adapter_path,
        "model_family": resolved_family,
        "model_name": model_name,
        "benchmarks": benchmarks,
        "mode": mode,
        "use_vllm": effective_use_vllm,
        "max_samples": max_samples,
        "gpu_id": gpu_id,
        "api_nproc": api_nproc,
    }
    (work_dir / "eval_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log.info(
        "eval_start",
        job_id=job_id,
        model_path=adapter_path,
        model_family=resolved_family,
        benchmarks=benchmarks,
    )

    try:
        vlmeval_dir = _resolve_vlmeval_dir(eval_cfg, repo_root)
        vlmeval_python = _resolve_vlmeval_python(eval_cfg)
        driver_path = work_dir / "eval_driver.py"
        _write_driver_script(
            script_path=driver_path,
            vlmeval_dir=vlmeval_dir,
            model_path=adapter_path,
            model_name=model_name,
            benchmarks=benchmarks,
            output_dir=results_dir,
            max_samples=max_samples,
            mode=mode,
            model_family=resolved_family,
            use_vllm=effective_use_vllm,
            api_nproc=api_nproc,
        )

        proc_env = os.environ.copy()
        proc_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Ensure VLMEvalKit writes under this run's eval job directory.
        proc_env["MMEVAL_ROOT"] = str(results_dir)
        if effective_use_vllm:
            proc_env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        proc = subprocess.run(
            [vlmeval_python, str(driver_path)],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            cwd=vlmeval_dir,
            env=proc_env,
        )

        stdout_path = work_dir / "vlmeval.stdout.log"
        stderr_path = work_dir / "vlmeval.stderr.log"
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")

        if proc.returncode != 0:
            error_output = (proc.stderr or proc.stdout or "")[-2000:]
            return json.dumps(
                {
                    "job_id": job_id,
                    "status": "failed",
                    "results_dir": str(results_dir),
                    "stdout_log": str(stdout_path),
                    "stderr_log": str(stderr_path),
                    "error": f"VLMEvalKit exited {proc.returncode}: {error_output}",
                }
            )

        metrics = _collect_metrics(results_dir, model_name, benchmarks)
        accuracy = float(sum(metrics.values()) / len(metrics)) if metrics else 0.0
        if mode == "all" and not metrics:
            error_tail = (proc.stderr or proc.stdout or "")[-1200:]
            discovered_files = _list_result_files(results_dir)
            return json.dumps(
                {
                    "job_id": job_id,
                    "status": "failed",
                    "results_dir": str(results_dir),
                    "stdout_log": str(stdout_path),
                    "stderr_log": str(stderr_path),
                    "discovered_files": discovered_files,
                    "error": f"VLMEvalKit completed but no parsable metrics were found. {error_tail}",
                }
            )

        payload = json.dumps(
            {
                "job_id": job_id,
                "model_family": resolved_family,
                "status": "completed",
                "results": metrics,
                "accuracy": accuracy,
            }
        )
        return StepResult(
            payload=payload,
            created=[ArtifactRef("eval_result", job_id, 0)],
            updates={"active_eval": ArtifactRef("eval_result", job_id, 0)},
            metrics={"job_id": job_id, "benchmarks": metrics},
        )

    except Exception as exc:
        log.exception("eval_failed", job_id=job_id)
        return json.dumps(
            {
                "job_id": job_id,
                "status": "failed",
                "error": str(exc),
            }
        )
