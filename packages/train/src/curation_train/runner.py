"""Multimodal finetune execution logic for Qwen and LLaVA methods."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from curation_train.defaults import (
    DEFAULT_LLAVA_MODEL,
    DEFAULT_QWEN_MODEL,
    LLAVA_FIXED_TRAINING_ARGS,
    QWEN_FIXED_TRAINING_ARGS,
)

logger = logging.getLogger(__name__)

_PKG_DIR = Path(__file__).parent
_TIMEOUT_SECONDS = 7200
_ALLOWED_OVERRIDES = {"max_steps", "save_steps", "logging_steps"}


def _resolve_deepspeed_config(name: str) -> str:
    """Resolve a deepspeed config name to its bundled path."""
    if not name.endswith(".json"):
        name = f"{name}.json"
    path = _PKG_DIR / "deepspeed" / name
    if not path.exists():
        raise FileNotFoundError(
            f"DeepSpeed config not found: {path}. "
            f"Available: {[p.name for p in (_PKG_DIR / 'deepspeed').glob('*.json')]}"
        )
    return str(path)


def _resolve_method(raw: object) -> str:
    method = str(raw or "qwen").lower().strip()
    if method in {"qwen", "qwen2.5", "qwen25"}:
        return "qwen"
    if method in {"llava", "llava1.5", "llava-1.5"}:
        return "llava"
    raise ValueError(f"Unsupported training method '{method}'. Use 'qwen' or 'llava'.")


def run_finetune(config: dict[str, Any]) -> dict[str, Any]:
    """Run multimodal finetune via DeepSpeed.

    Config JSON schema:
        method: str                 — qwen (default) or llava
        data_path: str              — path to Arrow dataset on disk
        output_dir: str             — where to write model outputs
        model_name_or_path: str     — base model (method-dependent default)
        deepspeed_config: str       — zero2/zero3/zero3_offload

    Returns:
        dict with keys: status, metrics, error
    """
    try:
        method = _resolve_method(config.get("method", "qwen"))
        data_path = config["data_path"]
        output_dir = config["output_dir"]

        if method == "qwen":
            model_name = config.get("model_name_or_path", DEFAULT_QWEN_MODEL)
            ds_config_name = config.get("deepspeed_config", "zero3")
            train_script = str(_PKG_DIR / "train_qwen2vl.py")
            fixed_args = dict(QWEN_FIXED_TRAINING_ARGS)
        else:
            model_name = config.get("model_name_or_path", DEFAULT_LLAVA_MODEL)
            ds_config_name = config.get("deepspeed_config", "zero2")
            train_script = str(_PKG_DIR / "train_llava15.py")
            fixed_args = dict(LLAVA_FIXED_TRAINING_ARGS)

        for key in _ALLOWED_OVERRIDES:
            if key in config:
                fixed_args[key] = config[key]

        ds_config_path = _resolve_deepspeed_config(ds_config_name)

        # Write the config JSON that the training script reads
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        train_config = {
            "method": method,
            "data_path": data_path,
            "model_name_or_path": model_name,
        }
        if method == "llava":
            train_config["lora_enable"] = bool(config.get("lora_enable", True))
            for key in ("lora_r", "lora_alpha", "lora_dropout", "lora_target_modules"):
                if key in config:
                    train_config[key] = config[key]
        train_config_path = str(Path(output_dir) / "train_config.json")
        Path(train_config_path).write_text(json.dumps(train_config, ensure_ascii=False))

        # Build DeepSpeed command with fixed training args
        cmd = [
            sys.executable,
            "-m",
            "deepspeed.launcher.runner",
            train_script,
            "--config_json", train_config_path,
            "--output_dir", output_dir,
            "--deepspeed", ds_config_path,
        ]

        for key, value in fixed_args.items():
            flag = f"--{key}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

        logger.info("Running: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )

        # Check for saved model as the success signal — deepspeed
        # can exit non-zero (e.g. SIGBUS from ZeRO-3 rank cleanup)
        # even after training and model save complete successfully.
        output_path = Path(output_dir)
        model_saved = (
            (output_path / "model.safetensors").exists()
            or any(output_path.glob("model-*.safetensors"))
            or (output_path / "adapter_model.safetensors").exists()
            or (output_path / "pytorch_model.bin").exists()
        )

        if result.returncode != 0 and not model_saved:
            raise RuntimeError(
                f"deepspeed exited with code {result.returncode}:\n"
                f"{(result.stderr or result.stdout)[-2000:]}"
            )

        if not model_saved:
            raise RuntimeError(
                f"Training process exited 0 but no model found in {output_dir}"
            )

        # Parse metrics from trainer output
        metrics: dict[str, Any] = {
            "method": method,
            "model_name_or_path": model_name,
            "output_dir": output_dir,
        }

        trainer_state = output_path / "trainer_state.json"
        if trainer_state.exists():
            state = json.loads(trainer_state.read_text())
            log_history = state.get("log_history", [])
            if log_history:
                last = log_history[-1]
                if "train_loss" in last:
                    metrics["train_loss"] = last["train_loss"]
                if "train_runtime" in last:
                    metrics["train_runtime"] = last["train_runtime"]

        return {
            "status": "completed",
            "metrics": metrics,
            "error": None,
        }

    except Exception as exc:
        logger.exception("Finetune failed")
        return {
            "status": "failed",
            "metrics": {},
            "error": str(exc),
        }
