"""Qwen2.5-VL-7B full finetune training script.

Launched via DeepSpeed. Loads an Arrow dataset (images + texts columns),
converts to Qwen2.5-VL ChatML format, and trains with HF Trainer.

Usage:
    deepspeed train_qwen2vl.py --config_json path/to/config.json
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    config_json: str = field(metadata={"help": "Path to JSON config file"})


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class QwenVLDataset(Dataset):
    """Wraps an Arrow dataset for Qwen2.5-VL training.

    Each example has:
        - images: PIL Image
        - texts: dict with "user" and "assistant" keys
    """

    def __init__(self, arrow_dataset, processor: AutoProcessor) -> None:
        self.dataset = arrow_dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.dataset[idx]
        images = example["images"]
        texts = example["texts"]

        if isinstance(texts, str):
            texts = json.loads(texts)

        # Normalize to list of turns
        if isinstance(texts, dict):
            texts = [texts]

        # Normalize images to list
        if not isinstance(images, list):
            images = [images]

        # Build multi-turn messages. First user turn gets all images.
        messages: list[dict[str, Any]] = []
        for i, turn in enumerate(texts):
            user_text = turn.get("user", turn.get("human", ""))
            assistant_text = turn.get("assistant", turn.get("gpt", ""))

            user_content: list[dict[str, Any]] = []
            if i == 0:
                for img in images:
                    user_content.append({"type": "image", "image": img})
            user_content.append({"type": "text", "text": user_text})

            messages.append({"role": "user", "content": user_content})
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            })

        # Apply chat template to get the full text
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Process vision info to get pixel values
        image_inputs, video_inputs = process_vision_info(messages)

        # Tokenize
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )

        # Squeeze batch dimension (processor returns batch dim)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        # Build labels: mask everything except assistant response with -100
        labels = input_ids.clone()
        labels = _mask_non_assistant_tokens(labels, self.processor.tokenizer)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if pixel_values is not None:
            result["pixel_values"] = pixel_values.squeeze(0) if pixel_values.dim() > 2 else pixel_values
        if image_grid_thw is not None:
            result["image_grid_thw"] = image_grid_thw

        return result


def _mask_non_assistant_tokens(
    labels: torch.Tensor, tokenizer
) -> torch.Tensor:
    """Mask all tokens that are not part of the assistant's response.

    Finds the last occurrence of the assistant header tokens
    (<|im_start|>assistant\\n) and masks everything before it.
    Also masks the trailing <|im_end|> token.
    """
    # Encode the assistant header marker
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Find the last <|im_start|> followed by assistant role
    # The pattern is: <|im_start|> assistant \n ... <|im_end|>
    assistant_token_ids = tokenizer.encode("assistant\n", add_special_tokens=False)

    label_list = labels.tolist()
    last_assistant_start = -1

    for i in range(len(label_list) - len(assistant_token_ids)):
        if label_list[i] == im_start_id:
            # Check if followed by "assistant\n"
            match = True
            for j, tid in enumerate(assistant_token_ids):
                if i + 1 + j >= len(label_list) or label_list[i + 1 + j] != tid:
                    match = False
                    break
            if match:
                last_assistant_start = i

    if last_assistant_start >= 0:
        # Mask everything up to and including the assistant header
        header_len = 1 + len(assistant_token_ids)  # <|im_start|> + "assistant\n"
        mask_end = last_assistant_start + header_len
        labels[:mask_end] = -100

        # Mask trailing <|im_end|> and any tokens after it
        for i in range(len(label_list) - 1, mask_end - 1, -1):
            if label_list[i] == im_end_id:
                labels[i:] = -100
                break
    else:
        # Fallback: if we can't find the pattern, mask everything
        # (this sample won't contribute to the loss)
        labels[:] = -100

    return labels


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------


class DataCollatorForQwenVL:
    """Pads input_ids, labels, attention_mask. Stacks pixel_values and image_grid_thw."""

    def __init__(self, processor: AutoProcessor) -> None:
        self.pad_token_id = processor.tokenizer.pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # Find max length for padding
        max_len = max(f["input_ids"].shape[0] for f in features)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []
        image_grid_thw_list = []

        for f in features:
            seq_len = f["input_ids"].shape[0]
            pad_len = max_len - seq_len

            # Pad on the right
            input_ids_list.append(
                torch.nn.functional.pad(f["input_ids"], (0, pad_len), value=self.pad_token_id)
            )
            attention_mask_list.append(
                torch.nn.functional.pad(f["attention_mask"], (0, pad_len), value=0)
            )
            labels_list.append(
                torch.nn.functional.pad(f["labels"], (0, pad_len), value=-100)
            )

            if "pixel_values" in f:
                pixel_values_list.append(f["pixel_values"])
            if "image_grid_thw" in f:
                image_grid_thw_list.append(f["image_grid_thw"])

        batch = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
        }

        if pixel_values_list:
            batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        if image_grid_thw_list:
            batch["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)

        return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def train() -> None:
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))

    # Parse --config_json first to load the JSON config
    # DeepSpeed passes extra args, so we use parse_args_into_dataclasses
    script_args, training_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=False
    )

    with open(script_args.config_json) as f:
        config = json.load(f)

    data_path = config["data_path"]
    model_name = config.get("model_name_or_path", "Qwen/Qwen2.5-VL-7B-Instruct")

    logger.info("Loading model: %s", model_name)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    processor = AutoProcessor.from_pretrained(model_name)

    # Ensure pad token is set
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    logger.info("Loading dataset from: %s", data_path)
    arrow_dataset = load_from_disk(data_path)

    train_dataset = QwenVLDataset(arrow_dataset, processor)

    collator = DataCollatorForQwenVL(processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()

    # Save model + processor
    output_dir = training_args.output_dir
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    logger.info("Training complete. Model saved to %s", output_dir)


if __name__ == "__main__":
    train()
