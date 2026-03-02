"""Fixed training configuration for multimodal finetune methods.

All hyperparameters are constants by default. The caller may provide a
small set of safe overrides in the run config (for smoke tests).
"""

DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_LLAVA_MODEL = "llava-hf/llava-1.5-7b-hf"

QWEN_FIXED_TRAINING_ARGS = dict(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    weight_decay=0.1,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    save_strategy="no",
    report_to="none",
    remove_unused_columns=False,
)

LLAVA_FIXED_TRAINING_ARGS = dict(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    weight_decay=0.0,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
    dataloader_num_workers=2,
    save_strategy="no",
    report_to="none",
    remove_unused_columns=False,
    logging_steps=1,
)
