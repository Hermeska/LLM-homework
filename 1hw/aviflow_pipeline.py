"""Two-step Aviflow pipeline for the first LLM homework."""

import argparse
import importlib
from pathlib import Path

import aviflow
from aviflow import Dataset, HTML, Input, Metrics, Model, Output, pipeline, remote


_aviflow_remote_module = importlib.import_module("aviflow.remote")
_aviflow_remote_module.PYPI_INDEX_URLS = [
    "https://download.pytorch.org/whl/cu128",
    "https://pypi.org/simple",
]

BASE_COMPONENT = "pytorch-dl/pytorch-train-dl:0.2.4"
RUNTIME_PACKAGES = [
    "transformers==4.52.0",
    "datasets==3.6.0",
    "accelerate==1.6.0",
    "matplotlib==3.10.1",
]
PROFILE_NAMES = (
    "baseline",
    "batch-accum",
    "large-batch",
    "compile-fused",
    "fp32-control",
    "optimized",
)


def prepare_wikipedia(
    source_code: str,
    tokenized_dataset: Output[Dataset],
) -> int:
    """Download and tokenize  Wikipedia into 32 parquet shards."""

    import pathlib

    namespace = {"__name__": "homework_solution"}
    exec(compile(source_code, "your_solution.py", "exec"), namespace)
    output_dir = pathlib.Path(tokenized_dataset.path)
    output_dir.mkdir(parents=True, exist_ok=True)
    namespace["prepare_dataset"](output_dir=str(output_dir))
    parquet_count = len(list(output_dir.glob("*.parquet")))
    if parquet_count != namespace["NUM_SHARDS"]:
        raise RuntimeError(
            f"Expected {namespace['NUM_SHARDS']} parquet shards, got {parquet_count}"
        )
    tokenized_dataset.metadata["shards"] = parquet_count
    tokenized_dataset.metadata["dataset"] = "wikimedia/wikipedia:20231101.ru"
    return parquet_count


def pretrain_qwen3(
    source_code: str,
    profile_name: str,
    tokenized_dataset: Input[Dataset],
    trained_model: Output[Model],
    final_metrics: Output[Metrics],
    experiment_report: Output[HTML],
) -> float:
    import base64
    import html
    import pathlib
    import tarfile

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Aviflow started the training step without a CUDA GPU")
    print(
        "GPU runtime:",
        f"device={torch.cuda.get_device_name(0)}",
        f"torch={torch.__version__}",
        f"cuda={torch.version.cuda}",
    )

    namespace = {"__name__": "homework_solution"}
    exec(compile(source_code, "your_solution.py", "exec"), namespace)
    output_dir = pathlib.Path("/tmp/llm-homework-output")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = namespace["train_model"](
        data_dir=tokenized_dataset.path,
        training_output_dir=str(output_dir),
        profile_name=profile_name,
    )

    final_metrics.log_metric("eval_loss", float(result["eval_loss"]))
    final_metrics.log_metric("perplexity", float(result["perplexity"]))
    final_metrics.log_metric("global_step", float(result["global_step"]))
    if "train_steps_per_second" in result["train_metrics"]:
        final_metrics.log_metric(
            "train_steps_per_second",
            float(result["train_metrics"]["train_steps_per_second"]),
        )

    model_path = pathlib.Path(trained_model.path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(model_path, "w:gz") as archive:
        archive.add(output_dir, arcname="qwen3-1b-russian")
    trained_model.framework = "transformers"
    trained_model.metadata["architecture"] = "Qwen3ForCausalLM"
    trained_model.metadata["profile"] = profile_name
    trained_model.metadata["trained_seconds"] = namespace[
        "MAX_TRAINING_TIME_SECONDS"
    ]

    plot_path = result.get("plot_path")
    image_html = ""
    if plot_path and pathlib.Path(plot_path).exists():
        encoded = base64.b64encode(pathlib.Path(plot_path).read_bytes()).decode()
        image_html = (
            '<img alt="Training loss" style="max-width:100%" '
            f'src="data:image/png;base64,{encoded}">'
        )
    sample_items = "".join(
        "<li><strong>"
        + html.escape(sample["prompt"])
        + "</strong><pre style='white-space:pre-wrap'>"
        + html.escape(sample["text"])
        + "</pre></li>"
        for sample in result["samples"]
    )
    #just report in HTML
    report = f"""<!doctype html>
<html lang="ru"><meta charset="utf-8"><title>LLM homework report</title>
<body style="font-family:system-ui;max-width:1000px;margin:40px auto;line-height:1.5">
<h1>Qwen3 mini-pretrain: Russian Wikipedia</h1>
<p><b>Experiment profile:</b> {html.escape(profile_name)}</p>
<p><b>Training:</b> 15 minutes on one GPU; sequence length 512; validation is the first 5,000 rows.</p>
<p><b>Final eval_loss:</b> {float(result['eval_loss']):.6f}<br>
<b>Perplexity:</b> {float(result['perplexity']):.4f}<br>
<b>Optimizer steps:</b> {int(result['global_step'])}<br>
<b>Train steps/second:</b> {float(result['train_metrics'].get('train_steps_per_second', 0.0)):.4f}</p>
<h2>Loss</h2>{image_html}
<h2>Generation samples</h2><ul>{sample_items}</ul>
</body></html>"""
    repor=t_path = pathlib.Path(experiment_report.path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    experiment_report.metadata["title"] = "Qwen3 Russian mini-pretrain"
    return float(result["eval_loss"])


def make_pipeline(run_name: str):
    prepare_wikipedia_task = remote(
        runtime_env={"base_component": BASE_COMPONENT, "pip": RUNTIME_PACKAGES},
        cpus=16,
        memory_mb=65536,
        enable_caching=True,
    )(prepare_wikipedia)
    pretrain_qwen3_task = remote(
        runtime_env={"base_component": BASE_COMPONENT, "pip": RUNTIME_PACKAGES},
        cpus=8,
        memory_mb=65536,
        shared_memory_mb=16384,
        gpus=1,
        enable_caching=False,
    )(pretrain_qwen3)

    @pipeline(experiment_name="llm-homework-1", run_name=run_name)
    def homework_pipeline(source_code: str, profile_name: str):
        prepared = prepare_wikipedia_task(source_code=source_code)
        pretrain_qwen3_task(
            source_code=source_code,
            profile_name=profile_name,
            tokenized_dataset=prepared.outputs["tokenized_dataset"],
        )

    return homework_pipeline


def main():
    parser = argparse.ArgumentParser(description="Launch homework in Aviflow")
    parser.add_argument("--namespace", default="students")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--profile",
        choices=PROFILE_NAMES,
        default="optimized",
    )
    parser.add_argument("--compile", metavar="PATH")
    args = parser.parse_args()

    source_code = Path(__file__).with_name("your_solution.py").read_text(
        encoding="utf-8"
    )
    aviflow.init(args.namespace)
    run_name = args.run_name or f"qwen3-1b-russian-{args.profile}"
    homework_pipeline = make_pipeline(run_name)
    parameters = {"source_code": source_code, "profile_name": args.profile}
    if args.compile:
        homework_pipeline.compile(path=args.compile, parameters=parameters)
    else:
        homework_pipeline.remote(parameters=parameters)


if __name__ == "__main__":
    main()
