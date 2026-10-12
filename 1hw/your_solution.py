import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 15
MAX_LENGTH = 512
INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
LABELS = "labels"

# Don't change these parameters
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000

TRAINING_CONFIG = {
    "output_dir": f"{OUTPUT_DIR}/gpt2-1b-russian",
    "optim": "adamw_torch_fused",
    "num_train_epochs": 1,
    "max_steps": 10000,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "learning_rate": 3e-4,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.1,
    "warmup_steps": 100,
    "max_grad_norm": 1.0,
    "logging_steps": 5,
    "eval_strategy": "no",
    "save_strategy": "no",
    "save_total_limit": 1,
    "load_best_model_at_end": False,
    "bf16": True,
    "tf32": True,
    "gradient_checkpointing": False,
    "dataloader_num_workers": 8,
    "dataloader_pin_memory": True,
    "torch_compile": True,
    "report_to": "none",
    "seed": 42,
    "data_seed": 42,
}


TRAINING_PROFILES = {
    "baseline": {
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "lr_scheduler_type": "linear",
        "optim": "adamw_torch",
        "torch_compile": False,
        "bf16": True,
        "tf32": False,
    },
    "batch-accum": {
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 2,
        "learning_rate": 5e-5,
        "lr_scheduler_type": "linear",
        "optim": "adamw_torch",
        "torch_compile": False,
        "bf16": True,
        "tf32": False,
    },
    "large-batch": {
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "lr_scheduler_type": "linear",
        "optim": "adamw_torch",
        "torch_compile": False,
        "bf16": True,
        "tf32": False,
    },
    "compile-fused": {
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "lr_scheduler_type": "linear",
        "optim": "adamw_torch_fused",
        "torch_compile": True,
        "bf16": True,
        "tf32": True,
    },
    "fp32-control": {
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "lr_scheduler_type": "linear",
        "optim": "adamw_torch",
        "torch_compile": False,
        "bf16": False,
        "tf32": False,
    },
    "optimized": {
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 3e-4,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch_fused",
        "torch_compile": True,
        "bf16": True,
        "tf32": True,
    },
}


class TimeoutCallback(TrainerCallback):
    """Stop training after the fixed homework time limit"""

    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                control.should_training_stop = True
                # Evaluate and save exactly the weights present at the timeout.
                control.should_evaluate = True
                control.should_save = True
                print(f"Training stopped after {elapsed:.2f} seconds")
        return control


def prepare_tokenizer():
    """Load the required tokenizer and make padding well-defined"""

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_function(examples, tokenizer):
    """Tokenize Wikipedia rows into fixed-length causal-LM samples"""

    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    tokenized[LABELS] = [
        [token_id if mask else -100 for token_id, mask in zip(ids, masks)]
        for ids, masks in zip(tokenized[INPUT_IDS], tokenized[ATTENTION_MASK])
    ]
    return {
        INPUT_IDS: tokenized[INPUT_IDS],
        ATTENTION_MASK: tokenized[ATTENTION_MASK],
        LABELS: tokenized[LABELS],
    }


def save_as_parquets(ds, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    """Save a dataset as deterministic, contiguous parquet shards."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for index in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=index, contiguous=True)
        shard_path = output_path / f"{index:05d}.parquet"
        print(f"Saving shard {index + 1}/{num_shards}: {shard_path}")
        shard.to_parquet(str(shard_path))


def prepare_dataset(output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    """Download, tokenize, and shard the required  Wikipedia dataset"""

    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")
    tokenizer = prepare_tokenizer()
    available_cpus = os.cpu_count() or 1
    num_proc = max(1, min(available_cpus, 16))
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing Russian Wikipedia",
    )
    save_as_parquets(tokenized_dataset, output_dir, num_shards)
    return str(Path(output_dir).resolve())


def load_tokenized_dataset(data_dir=OUTPUT_DIR):
    """Load only sorted parquet shards from a prepared dataset directory."""

    parquet_files = sorted(str(path) for path in Path(data_dir).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet shards found in {data_dir}. Run data preparation first"
        )
    return load_dataset("parquet", data_files=parquet_files)["train"]


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    """Keep the first 5,000 rows for validation exactly as required"""

    dataset_size = len(dataset)
    if dataset_size <= validation_size:
        raise ValueError(
            f"Dataset has {dataset_size} rows; it must exceed {validation_size}."
        )
    eval_dataset = dataset.select(range(validation_size))
    train_dataset = dataset.select(range(validation_size, dataset_size))

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def create_model(tokenizer, torch_dtype=torch.bfloat16):
    # Don't change this parameter
    MODEL_CONFIG = {
        "hidden_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 8192,
        "head_dim": 128,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "scale_attn_weights": True,
        "use_cache": True,
    }

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG,
    )
    #attention brrr
    if torch_dtype == torch.float32:
        attention_implementation = "sdpa"
        print("FP32 profile selected; using PyTorch SDPA")
    else:
        try:
            import flash_attn  # noqa: F401

            attention_implementation = "flash_attention_2"
        except ImportError:
            attention_implementation = "sdpa"
            print("flash-attn is unavailable; using PyTorch SDPA")

    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation=attention_implementation,
        torch_dtype=torch_dtype,
    )

    print(f"Attention implementation: {attention_implementation}")
    print(f"Model pad token id: {model.config.pad_token_id}")
    with torch.no_grad():
        total_params = sum(parameter.numel() for parameter in model.parameters())
        print(f"Total params: {total_params:,}")
    return model


def _latest_evaluation(log_history):
    evaluations = [row for row in log_history if "eval_loss" in row]
    return evaluations[-1] if evaluations else None


def _save_history(log_history, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history_path = output_path / "training_history.csv"
    columns = sorted({key for row in log_history for key in row})
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(log_history)

    losses = [row for row in log_history if "loss" in row]
    if not losses:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values = [row.get("step", index) for index, row in enumerate(losses)]
    y_values = [row["loss"] for row in losses]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(x_values, y_values, linewidth=1.5, color="#2563eb")
    axis.set_title("Training loss during the 15-minute pretrain")
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    plot_path = output_path / "training_loss.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return str(plot_path)


def _generate_samples(model, tokenizer):
    prompts = [
        "Москва — столица",
        "Искусственный интеллект помогает человеку",
        "Научное открытие начинается с",
    ]
    model.eval()
    device = next(model.parameters()).device
    samples = []
    for index, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        torch.manual_seed(42 + index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(42 + index)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        samples.append(
            {
                "prompt": prompt,
                "text": tokenizer.decode(output_ids[0], skip_special_tokens=True),
            }
        )
    return samples


def _write_experiment_summary(
    output_dir,
    profile_name,
    config,
    train_metrics,
    eval_results,
    global_step,
    samples,
):
    eval_loss = float(eval_results["eval_loss"])
    perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
    #template
    lines = [
        "# Mini-pretrain report",
        "",
        "## Setup",
        "",
        f"- Experiment profile: {profile_name}",
        f"- Model: Qwen3 from scratch, {config['model_parameter_count']:,} parameters",
        "- Dataset: wikimedia/wikipedia, 20231101.ru",
        f"- Validation: first {VALIDATION_SIZE} samples",
        f"- Sequence length: {MAX_LENGTH}",
        f"- Training time limit: {MAX_TRAINING_TIME_SECONDS // 60} minutes",
        f"- Batch size: {config['per_device_train_batch_size']}",
        f"- Gradient accumulation: {config['gradient_accumulation_steps']}",
        f"- Effective batch size: {config['effective_batch_size']}",
        f"- Learning rate: {config['learning_rate']}",
        f"- Scheduler: {config['lr_scheduler_type']}",
        f"- Optimizer: {config['optim']}",
        f"- torch.compile: {config['torch_compile']}",
        f"- bf16: {config['bf16']}",
        f"- tf32: {config['tf32']}",
        "",
        "## Final metrics",
        "",
        f"- optimizer steps: {global_step}",
        f"- train runtime: {float(train_metrics.get('train_runtime', 0.0)):.2f} seconds",
        f"- train steps/second: {float(train_metrics.get('train_steps_per_second', 0.0)):.4f}",
        f"- eval_loss: {eval_loss:.6f}",
        f"- perplexity: {perplexity:.4f}" if math.isfinite(perplexity) else "- perplexity: inf",
        "",
        "## Generations",
        "",
    ]
    for sample in samples:
        lines.extend(
            [
                f"### Prompt: {sample['prompt']}",
                "",
                sample["text"],
                "",
            ]
        )
    lines.extend(
        [
            "## Observations",
            "",
            "Add a short comparison with later hyperparameter runs here.",
        ]
    )
    Path(output_dir, "experiment_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def train_model(
    data_dir=OUTPUT_DIR,
    training_output_dir=None,
    profile_name="optimized",
):
    """Run train"""

    if profile_name not in TRAINING_PROFILES:
        raise ValueError(
            f"Unknown profile {profile_name!r}; choose one of "
            f"{sorted(TRAINING_PROFILES)}"
        )

    tokenizer = prepare_tokenizer()
    dataset = load_tokenized_dataset(data_dir)
    train_dataset, eval_dataset = split_dataset(dataset)

    training_config = dict(TRAINING_CONFIG)
    training_config.update(TRAINING_PROFILES[profile_name])
    default_output_dir = f"{TRAINING_CONFIG['output_dir']}-{profile_name}"
    output_dir = training_output_dir or default_output_dir
    training_config["output_dir"] = str(output_dir)
    model_dtype = torch.bfloat16 if training_config["bf16"] else torch.float32
    model = create_model(tokenizer, torch_dtype=model_dtype)
    print(f"Experiment profile: {profile_name}")
    print(json.dumps(training_config, indent=2, default=str))
    training_args = TrainingArguments(**training_config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS)],  # dont change
    )

    train_result = trainer.train()
    final_evaluation = _latest_evaluation(trainer.state.log_history)
    if final_evaluation is None:
        print("Running final evaluation...")
        final_evaluation = trainer.evaluate()
    else:
        print("Using the evaluation requested at the 15-minute timeout.")
    print(f"Final evaluation results: {final_evaluation}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_state()
    trainer.save_model(str(output_path / "final_model"))
    tokenizer.save_pretrained(str(output_path / "final_model"))
    Path(output_path, "evaluation_results.json").write_text(
        json.dumps(final_evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_path = _save_history(trainer.state.log_history, output_path)
    samples = _generate_samples(model, tokenizer)
    Path(output_path, "generation_samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_config = dict(training_config)
    report_config["model_parameter_count"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    report_config["effective_batch_size"] = (
        training_config["per_device_train_batch_size"]
        * training_config["gradient_accumulation_steps"]
    )
    _write_experiment_summary(
        output_path,
        profile_name,
        report_config,
        train_result.metrics,
        final_evaluation,
        trainer.state.global_step,
        samples,
    )

    eval_loss = float(final_evaluation["eval_loss"])
    return {
        "eval_results": final_evaluation,
        "eval_loss": eval_loss,
        "perplexity": math.exp(eval_loss) if eval_loss < 20 else float("inf"),
        "samples": samples,
        "plot_path": plot_path,
        "output_dir": str(output_path.resolve()),
        "global_step": trainer.state.global_step,
        "train_metrics": train_result.metrics,
        "profile_name": profile_name,
        "training_config": training_config,
    }


def main():
    parser = argparse.ArgumentParser(description="Russian Qwen3 mini-pretrain")
    parser.add_argument(
        "mode",
        choices=("prepare", "train", "all"),
        nargs="?",
        default="train",
        help="prepare parquet data, train from prepared data, or do both",
    )
    parser.add_argument("--data-dir", default=OUTPUT_DIR)
    parser.add_argument("--training-output-dir")
    parser.add_argument(
        "--profile",
        choices=sorted(TRAINING_PROFILES),
        default="optimized",
    )
    args = parser.parse_args()

    if args.mode in {"prepare", "all"}:
        prepare_dataset(output_dir=args.data_dir)
    if args.mode in {"train", "all"}:
        train_model(
            data_dir=args.data_dir,
            training_output_dir=args.training_output_dir,
            profile_name=args.profile,
        )


if __name__ == "__main__":
    main()
