import argparse
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)
from transformers.utils import ModelOutput  # pyright: ignore[reportPrivateImportUsage]
from trl import SFTConfig, SFTTrainer  # pyright: ignore[reportPrivateImportUsage]
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

DATA: Path = Path("data")
TRAIN_SECONDS: float = 300
ATTENTION_IMPLEMENTATION: str = "sdpa"

MODEL_CONFIG: dict[str, Any] = {
    "hidden_size": 2048,
    "num_hidden_layers": 12,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 8192,
    "head_dim": 128,
    "use_cache": False,
}

TRAINING_CONFIG: dict[str, Any] = {
    "output_dir": "results/baseline",
    # Настройки для экспериментов.
    "per_device_train_batch_size": 32,
    "gradient_accumulation_steps": 2,
    "gradient_checkpointing": False,
    "torch_compile": False,
    "use_liger_kernel": False,
    "activation_offloading": False,
    "packing": False,
    "padding_free": False,
    # Общие параметры обучения и замера.
    "max_steps": 1_000_000,
    "learning_rate": 3e-4,
    "lr_scheduler_type": "constant_with_warmup",
    "warmup_steps": 200,
    "weight_decay": 0.01,
    "optim": "adamw_torch_fused",
    "bf16": True,
    "tf32": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "packing_strategy": "bfd",
    "max_length": 512,
    "completion_only_loss": False,
    "loss_type": "nll",
    "dataloader_num_workers": 2,
    "dataloader_pin_memory": True,
    "logging_steps": 10,
    "logging_nan_inf_filter": False,
    "report_to": "none",
    "save_strategy": "no",
    "eval_strategy": "no",
    "seed": 42,
    "data_seed": 42,
    "disable_tqdm": True,
}


# These values are part of the homework protocol and must stay identical in
# every run. Experiment profiles are validated against this mapping below.
LOCKED_TRAINING_CONFIG: dict[str, Any] = {
    "max_length": 512,
    "bf16": True,
    "optim": "adamw_torch_fused",
    "learning_rate": 3e-4,
    "lr_scheduler_type": "constant_with_warmup",
    "warmup_steps": 200,
    "weight_decay": 0.01,
    "seed": 42,
    "data_seed": 42,
    "completion_only_loss": False,
    "loss_type": "nll",
}


# Only performance-related fields vary. Every profile is launched in a fresh
# process so global patches made by Liger/torch.compile cannot leak into the
# next experiment. The two combination profiles satisfy the requirement to
# evaluate interactions between mechanisms.
EXPERIMENT_PROFILES: dict[str, dict[str, Any]] = {
    "baseline": {
        "description": "Template baseline: dynamic padding and PyTorch SDPA.",
        "attention_implementation": "sdpa",
        "overrides": {},
    },
    "batch-64": {
        "description": "Same effective batch, fewer microbatches: batch 64 x accumulation 1.",
        "attention_implementation": "sdpa",
        "overrides": dict(
            per_device_train_batch_size=64,
            gradient_accumulation_steps=1,
        ),
    },
    "checkpointing": dict(
        description="Activation checkpointing trades recomputation for GPU memory.",
        attention_implementation="sdpa",
        overrides=dict(gradient_checkpointing=True),
    ),
    "packing": dict(
        description="Wrapped sequence packing reduces padding without padding-free attention.",
        attention_implementation="sdpa",
        overrides=dict(packing=True, packing_strategy="wrapped"),
    ),
    "compile": dict(
        description="PyTorch torch.compile with the otherwise unchanged baseline.",
        attention_implementation="sdpa",
        overrides=dict(torch_compile=True),
    ),
    "liger": dict(
        description="Liger fused kernels with the otherwise unchanged baseline.",
        attention_implementation="sdpa",
        overrides=dict(use_liger_kernel=True),
    ),
    "activation-offload": dict(
        description="Move saved activations to CPU to reduce peak GPU memory.",
        attention_implementation="sdpa",
        overrides=dict(activation_offloading=True),
    ),
    "packing-compile": dict(
        description="Combination: wrapped packing plus torch.compile.",
        attention_implementation="sdpa",
        overrides=dict(
            packing=True,
            packing_strategy="wrapped",
            torch_compile=True,
        ),
    ),
    "packing-liger": dict(
        description="Combination: wrapped packing plus Liger kernels.",
        attention_implementation="sdpa",
        overrides=dict(
            packing=True,
            packing_strategy="wrapped",
            use_liger_kernel=True,
        ),
    ),
}

SHORT_TRAIN_SECONDS: float = 60


def prepare_data() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "ai-forever/rugpt3small_based_on_gpt2",
        revision="a9307e696cd3c5b7f953ff4cb19d76a4d81821d5",
    )
    tokenizer.pad_token = tokenizer.eos_token
    source = load_dataset(
        "wikimedia/wikipedia",
        "20231101.ru",
        split="train",
        streaming=True,
        revision="b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    )
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for index, article in enumerate(source.take(50_500)):
        split = "validation" if index < 500 else "train"
        paragraphs = [
            p.strip() for p in article["text"].splitlines() if len(p.strip()) >= 80
        ][:8]
        if paragraphs:
            encoded = tokenizer(
                paragraphs,
                add_special_tokens=False,
                truncation=True,
                max_length=511,
                return_attention_mask=False,
            )["input_ids"]
            for paragraph_index, tokens in enumerate(encoded):
                if len(tokens) >= 32:
                    rows[split].append(
                        dict(
                            input_ids=tokens + [tokenizer.eos_token_id],
                            article_id=article["id"],
                            paragraph_index=paragraph_index,
                        )
                    )
        if (index + 1) % 5000 == 0:
            print(f"Подготовлено {index + 1}/50500 статей", flush=True)
    DatasetDict(
        {name: Dataset.from_list(items) for name, items in rows.items()}
    ).save_to_disk(str(DATA / "dataset"))
    tokenizer.save_pretrained(DATA / "tokenizer")


class CausalCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = super().torch_call(examples)
        batch["labels"][:, 0] = -100
        return batch


class BenchmarkTrainer(SFTTrainer):
    target_tokens: torch.Tensor

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ModelOutput]:
        return Trainer.compute_loss(
            self,
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.target_tokens += inputs["labels"].ne(-100).sum()
        return super().training_step(model, inputs, num_items_in_batch)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        for key in ("loss", "grad_norm", "train_loss"):
            if key in logs and not math.isfinite(logs[key]):
                raise FloatingPointError(f"Значение {key} не валидное")
        logs.pop("train_samples_per_second", None)
        logs.pop("train_steps_per_second", None)
        super().log(logs, start_time)


class Benchmark(TrainerCallback):
    start: float
    previous_end: float
    previous_tokens: int

    def __init__(self, trainer: BenchmarkTrainer, seconds: float) -> None:
        self.trainer, self.seconds = trainer, seconds
        self.steps: list[dict[str, float | int | bool]] = []

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.start = self.previous_end = time.perf_counter()
        self.previous_tokens = 0

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        torch.cuda.synchronize()
        now = time.perf_counter()
        tokens = self.trainer.target_tokens.item()
        record: dict[str, float | int | bool] = dict(
            step=state.global_step,
            elapsed_seconds=now - self.start,
            step_seconds=now - self.previous_end,
            target_tokens=tokens - self.previous_tokens,
            warmup=state.global_step <= 10,
        )
        self.steps.append(record)
        if state.global_step % 10 == 0:
            print(json.dumps(record), flush=True)
        self.previous_end, self.previous_tokens = now, tokens
        if now - self.start >= self.seconds:
            control.should_training_stop = True

    def summary(self) -> dict[str, float | int | None]:
        measured = self.steps[10:]
        duration = sum(step["step_seconds"] for step in measured)
        tokens = sum(step["target_tokens"] for step in measured)
        elapsed = self.previous_end - self.start
        return dict(
            train_seconds=elapsed,
            optimizer_steps=len(self.steps),
            target_tokens=self.previous_tokens,
            tokens_per_second_including_warmup=self.previous_tokens / elapsed,
            warmup_steps=10,
            measured_steps=len(measured),
            measured_seconds=duration,
            tokens_per_second=tokens / duration if measured else None,
            seconds_per_optimizer_step=duration / len(measured) if measured else None,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, float | int]:
    loader = DataLoader(
        dataset.select_columns(["input_ids"]),
        batch_size=8,
        collate_fn=CausalCollator(tokenizer.pad_token_id, completion_only_loss=False),
    )
    model.eval()
    loss_sum, tokens = 0.0, 0
    for batch in loader:
        batch = {key: value.cuda() for key, value in batch.items()}
        count = batch["labels"].ne(-100).sum().item()
        loss_sum += model(**batch).loss.item() * count
        tokens += count
    loss = loss_sum / tokens

    return dict(validation_loss=loss, validation_target_tokens=tokens)


def build_training_config(profile_name: str, output_dir: Path) -> dict[str, Any]:
    if profile_name not in EXPERIMENT_PROFILES:
        raise ValueError(
            f"Неизвестный профиль {profile_name!r}; выберите один из "
            f"{sorted(EXPERIMENT_PROFILES)}"
        )

    training_config = dict(TRAINING_CONFIG)
    training_config.update(EXPERIMENT_PROFILES[profile_name]["overrides"])
    training_config["output_dir"] = str(output_dir)

    for key, expected in LOCKED_TRAINING_CONFIG.items():
        actual = training_config[key]
        if actual != expected:
            raise ValueError(
                f"Профиль {profile_name!r} изменил обязательный параметр "
                f"{key}: ожидалось {expected!r}, получено {actual!r}"
            )
    return training_config


def run_experiment(
    profile_name: str,
    seconds: float,
    output_dir: Path,
    run_name: str | None = None,
) -> dict[str, Any]:
    training_config = build_training_config(profile_name, output_dir)
    training_args = SFTConfig(**training_config)

    output_dir.mkdir(parents=True, exist_ok=False)
    if not (DATA / "dataset").exists():
        prepare_data()
    tokenizer = AutoTokenizer.from_pretrained(DATA / "tokenizer")
    dataset = load_from_disk(str(DATA / "dataset"))

    set_seed(training_args.seed)
    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG,
    )
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation=EXPERIMENT_PROFILES[profile_name][
            "attention_implementation"
        ],
        torch_dtype=torch.bfloat16,
    )

    trainer = BenchmarkTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
        data_collator=(
            None
            if training_args.packing or training_args.padding_free
            else CausalCollator(tokenizer.pad_token_id)
        ),
    )
    trainer.target_tokens = torch.zeros((), dtype=torch.int64, device="cuda")
    benchmark = Benchmark(trainer, seconds)
    trainer.add_callback(benchmark)

    trainer.train()
    summary = benchmark.summary()
    summary.update(
        evaluate(trainer.model, dataset["validation"].select(range(512)), tokenizer)
    )
    summary.update(
        status="succeeded",
        run_name=run_name or profile_name,
        profile=profile_name,
        description=EXPERIMENT_PROFILES[profile_name]["description"],
        requested_train_seconds=seconds,
        attention_implementation=EXPERIMENT_PROFILES[profile_name][
            "attention_implementation"
        ],
        effective_batch_size=(
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
        ),
        training_config=training_config,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_single(
    profile_name: str,
    seconds: float,
    output_dir: Path,
    run_name: str | None = None,
) -> int:
    try:
        run_experiment(profile_name, seconds, output_dir, run_name=run_name)
        return 0
    except Exception as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = dict(
            status="failed",
            run_name=run_name or profile_name,
            profile=profile_name,
            requested_train_seconds=seconds,
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        (output_dir / "summary.json").write_text(
            json.dumps(failure, indent=2) + "\n"
        )
        traceback.print_exc()
        print(json.dumps(failure, indent=2), flush=True)
        return 1


def _launch_fresh_process(
    profile_name: str,
    seconds: float,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--profile",
        profile_name,
        "--train-seconds",
        str(seconds),
        "--output-dir",
        str(output_dir),
        "--data-dir",
        str(DATA),
        "--run-name",
        run_name,
    ]
    print(f"Launching: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, check=False)
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return dict(
            status="failed",
            run_name=run_name,
            profile=profile_name,
            requested_train_seconds=seconds,
            error=f"Process exited with code {completed.returncode} without summary.json",
        )
    return json.loads(summary_path.read_text())


def _metric(summary: dict[str, Any], key: str) -> str:
    value = summary.get(key)
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_suite_report(suite: dict[str, Any], output_root: Path) -> None:
    experiments = suite["experiments"]
    lines = [
        "# Homework 2: training efficiency report",
        "",
        "Все эксперименты запущены с нуля последовательно на одной GPU. ",
        "Модель, токенизатор, датасет, split, длина 512, BF16, optimizer, ",
        "learning rate, scheduler, seed, evaluation и подсчёт метрик не менялись.",
        "",
        "| Run | Profile | Seconds | Tokens/s | Peak allocated GiB | "
        "Validation loss | Optimizer steps | Status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in experiments:
        lines.append(
            "| {run_name} | {profile} | {seconds} | {tps} | {memory} | "
            "{loss} | {steps} | {status} |".format(
                run_name=item["run_name"],
                profile=item["profile"],
                seconds=_metric(item, "train_seconds"),
                tps=_metric(item, "tokens_per_second"),
                memory=_metric(item, "peak_allocated_gib"),
                loss=_metric(item, "validation_loss"),
                steps=_metric(item, "optimizer_steps"),
                status=item["status"],
            )
        )

    baseline = suite["baseline"]
    final = suite["final"]
    by_profile = {
        item["profile"]: item
        for item in experiments
        if item.get("status") == "succeeded"
        and item["run_name"].startswith("short-")
    }

    lines.extend(
        [
            "",
            "## Final comparison",
            "",
            f"- Baseline tokens/s: {baseline['tokens_per_second']:.4f}",
            f"- Final tokens/s: {final['tokens_per_second']:.4f}",
            f"- Speedup S: **{suite['speedup']:.4f}x**",
            f"- Selected profile: `{suite['selected_profile']}`",
            f"- Baseline validation loss: {baseline['validation_loss']:.4f}",
            f"- Final validation loss: {final['validation_loss']:.4f}",
            "",
            "## Observations",
            "",
            "1. Batch/accumulation changes the number of microbatches per optimizer "
            "step while keeping effective batch 64. Compare `batch-64` with baseline.",
            "2. Checkpointing and activation offloading reduce stored GPU activations, "
            "but add recomputation or CPU transfers. Compare speed and peak memory.",
            "3. Packing removes padding work. Its interactions are measured separately "
            "in `packing-compile` and `packing-liger`, so a combination is not assumed "
            "to equal the sum of individual improvements.",
        ]
    )

    if "packing" in by_profile and "packing-compile" in by_profile:
        packing = by_profile["packing"]["tokens_per_second"]
        combination = by_profile["packing-compile"]["tokens_per_second"]
        lines.append(
            f"4. Packing + compile versus packing alone: "
            f"{combination / packing:.4f}x ({combination:.2f} vs {packing:.2f} tokens/s)."
        )
    if "packing" in by_profile and "packing-liger" in by_profile:
        packing = by_profile["packing"]["tokens_per_second"]
        combination = by_profile["packing-liger"]["tokens_per_second"]
        lines.append(
            f"5. Packing + Liger versus packing alone: "
            f"{combination / packing:.4f}x ({combination:.2f} vs {packing:.2f} tokens/s)."
        )

    (output_root / "research_report.md").write_text("\n".join(lines) + "\n")


def run_suite(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    if not (DATA / "dataset").exists():
        prepare_data()

    experiments: list[dict[str, Any]] = []
    baseline = _launch_fresh_process(
        "baseline",
        TRAIN_SECONDS,
        output_root / "baseline-300s",
        "baseline-300s",
    )
    experiments.append(baseline)
    if baseline.get("status") != "succeeded":
        raise RuntimeError("Baseline failed; stopping the benchmark suite")

    candidate_profiles = [
        profile for profile in EXPERIMENT_PROFILES if profile != "baseline"
    ]
    for profile_name in candidate_profiles:
        run_name = f"short-{profile_name}-{int(SHORT_TRAIN_SECONDS)}s"
        experiments.append(
            _launch_fresh_process(
                profile_name,
                SHORT_TRAIN_SECONDS,
                output_root / run_name,
                run_name,
            )
        )

    successful_candidates = [
        item
        for item in experiments[1:]
        if item.get("status") == "succeeded"
        and item.get("tokens_per_second") is not None
    ]
    if len(successful_candidates) < 5:
        raise RuntimeError(
            "Fewer than five optimization candidates succeeded; cannot satisfy "
            "the six-configuration requirement together with baseline"
        )

    winner = max(successful_candidates, key=lambda item: item["tokens_per_second"])
    selected_profile = winner["profile"]
    final_run_name = f"final-{selected_profile}-300s"
    final = _launch_fresh_process(
        selected_profile,
        TRAIN_SECONDS,
        output_root / final_run_name,
        final_run_name,
    )
    experiments.append(final)
    if final.get("status") != "succeeded":
        raise RuntimeError("Final 300-second run failed")

    suite = dict(
        baseline=baseline,
        candidates=experiments[1:-1],
        selected_profile=selected_profile,
        final=final,
        speedup=final["tokens_per_second"] / baseline["tokens_per_second"],
        experiments=experiments,
        gpu_name=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
    )
    (output_root / "suite_summary.json").write_text(
        json.dumps(suite, indent=2) + "\n"
    )
    write_suite_report(suite, output_root)
    print(json.dumps(suite, indent=2), flush=True)
    return suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3 training efficiency benchmark")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--profile", choices=sorted(EXPERIMENT_PROFILES), default="baseline")
    parser.add_argument("--train-seconds", type=float, default=TRAIN_SECONDS)
    parser.add_argument("--output-dir")
    parser.add_argument("--data-dir", default=str(DATA))
    parser.add_argument("--run-name")
    return parser.parse_args()


def main() -> int:
    global DATA
    args = parse_args()
    DATA = Path(args.data_dir)
    if args.prepare_only:
        prepare_data()
        return 0
    if args.suite:
        output_root = Path(args.output_dir or "results/benchmark-suite")
        run_suite(output_root)
        return 0

    output_dir = Path(args.output_dir or TRAINING_CONFIG["output_dir"])
    return run_single(
        args.profile,
        args.train_seconds,
        output_dir,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    raise SystemExit(main())
