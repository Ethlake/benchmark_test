"""LLaVA-1.5 training entrypoint for curation-train.

Launched via DeepSpeed:
    deepspeed train_llava15.py --config_json path/to/config.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    LlavaForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    config_json: str = field(metadata={"help": "Path to JSON config file"})


def _as_pil_image(value: Any) -> Image.Image:
    """Convert a dataset image cell into PIL RGB."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if "bytes" in value and value["bytes"] is not None:
            from io import BytesIO

            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if "path" in value and value["path"]:
            return Image.open(value["path"]).convert("RGB")
    if hasattr(value, "convert"):
        return value.convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value)!r}")


def _normalize_turns(texts: Any) -> list[dict[str, str]]:
    if isinstance(texts, str):
        texts = json.loads(texts)
    if isinstance(texts, dict):
        texts = [texts]
    if not isinstance(texts, list):
        return [{"user": str(texts), "assistant": ""}]

    turns: list[dict[str, str]] = []
    for turn in texts:
        if isinstance(turn, dict):
            user = str(turn.get("user", turn.get("human", "")))
            assistant = str(turn.get("assistant", turn.get("gpt", "")))
            turns.append({"user": user, "assistant": assistant})
        else:
            turns.append({"user": str(turn), "assistant": ""})
    return turns or [{"user": "", "assistant": ""}]


class Llava15Dataset(Dataset):
    """Arrow dataset wrapper for LLaVA-1.5 HF training."""

    def __init__(self, arrow_dataset, processor: AutoProcessor) -> None:
        self.dataset = arrow_dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.dataset[idx]
        images = example.get("images")
        texts = example.get("texts")

        if isinstance(images, list):
            image = _as_pil_image(images[0])
        else:
            image = _as_pil_image(images)

        turns = _normalize_turns(texts)

        conversation: list[dict[str, Any]] = []
        for turn_idx, turn in enumerate(turns):
            user_content: list[dict[str, Any]] = []
            if turn_idx == 0:
                user_content.append({"type": "image"})
            user_content.append({"type": "text", "text": turn["user"]})
            conversation.append({"role": "user", "content": user_content})
            conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": turn["assistant"]}],
                }
            )

        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            tokenize=False,
        )

        model_inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        batch: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        for key, value in model_inputs.items():
            if key in {"input_ids", "attention_mask"}:
                continue
            batch[key] = value.squeeze(0) if hasattr(value, "squeeze") else value
        return batch


class DataCollatorForLlava15:
    """Pad token sequences and stack image tensors."""

    def __init__(self, processor: AutoProcessor) -> None:
        tokenizer = processor.tokenizer
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            pad_id = tokenizer.pad_token_id
        self.pad_token_id = int(pad_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(f["input_ids"].shape[0] for f in features)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for feature in features:
            seq_len = feature["input_ids"].shape[0]
            pad_len = max_len - seq_len
            input_ids_list.append(
                torch.nn.functional.pad(
                    feature["input_ids"], (0, pad_len), value=self.pad_token_id
                )
            )
            attention_mask_list.append(
                torch.nn.functional.pad(feature["attention_mask"], (0, pad_len), value=0)
            )
            labels_list.append(
                torch.nn.functional.pad(feature["labels"], (0, pad_len), value=-100)
            )

        batch: dict[str, Any] = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
        }

        extra_keys = sorted(
            {k for f in features for k in f.keys()} - {"input_ids", "attention_mask", "labels"}
        )
        for key in extra_keys:
            values = [f[key] for f in features if key in f]
            if len(values) != len(features):
                continue
            if all(torch.is_tensor(v) for v in values):
                try:
                    batch[key] = torch.stack(values)
                except RuntimeError:
                    batch[key] = values
            else:
                batch[key] = values

        return batch


def train() -> None:
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=False
    )

    config = json.loads(Path(script_args.config_json).read_text())
    data_path = config["data_path"]
    model_name = config.get("model_name_or_path", "llava-hf/llava-1.5-7b-hf")
    lora_enable = bool(config.get("lora_enable", True))

    logger.info("Loading model: %s", model_name)
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )
    model.config.use_cache = False

    if lora_enable:
        lora_cfg = LoraConfig(
            r=int(config.get("lora_r", 16)),
            lora_alpha=int(config.get("lora_alpha", 32)),
            lora_dropout=float(config.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(
                config.get(
                    "lora_target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj"],
                )
            ),
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    processor = AutoProcessor.from_pretrained(model_name)

    logger.info("Loading dataset from: %s", data_path)
    ds = load_from_disk(data_path)
    if isinstance(ds, DatasetDict):
        if "train" in ds:
            ds = ds["train"]
        else:
            ds = concatenate_datasets([ds[k] for k in sorted(ds.keys())])

    train_dataset = Llava15Dataset(ds, processor)
    collator = DataCollatorForLlava15(processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.train()

    output_dir = training_args.output_dir

    if lora_enable:
        logger.info("Merging LoRA adapter into base model...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)

    processor.save_pretrained(output_dir)

    # Patch tokenizer_config.json for compatibility with transformers 4.x
    # (used by VLMEvalKit's conda env). Transformers 5.x writes
    # "tokenizer_class": "TokenizersBackend" which 4.x doesn't recognize.
    tok_config_path = Path(output_dir) / "tokenizer_config.json"
    if tok_config_path.exists():
        tok_config = json.loads(tok_config_path.read_text())
        if tok_config.get("tokenizer_class") == "TokenizersBackend":
            tok_config.pop("tokenizer_class")
            tok_config.pop("model_specific_special_tokens", None)
            tok_config_path.write_text(json.dumps(tok_config, indent=2, ensure_ascii=False))

    logger.info("Training complete. Model saved to %s", output_dir)


if __name__ == "__main__":
    train()
