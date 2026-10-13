"""Run the complete Homework 2 benchmark suite on one Aviflow GPU."""

import argparse
import importlib
from pathlib import Path

import aviflow
from aviflow import Dataset, HTML, Input, Metrics, Output, pipeline, remote


_aviflow_remote_module = importlib.import_module("aviflow.remote")
_aviflow_remote_module.PYPI_INDEX_URLS = [
    "https://download.pytorch.org/whl/cu128",
    "https://pypi.org/simple",
]

BASE_COMPONENT = "pytorch-dl/pytorch-train-dl:0.2.4"
COMMON_PACKAGES = [
    "transformers==4.56.2",
    "datasets==3.6.0",
    "accelerate==1.6.0",
    "trl==0.23.1",
]
GPU_PACKAGES = COMMON_PACKAGES + ["liger-kernel==0.6.2"]


def prepare_benchmark_data(
    source_code: str,
    prepared_data: Output[Dataset],
) -> int:
    """Prepare the exact tokenizer, dataset, and train/validation split once"""

    import pathlib

    namespace = {"__name__": "homework2_solution"}
    exec(compile(source_code, "train.py", "exec"), namespace)
    data_dir = pathlib.Path(prepared_data.path)
    data_dir.mkdir(parents=True, exist_ok=True)
    namespace["DATA"] = data_dir
    namespace["prepare_data"]()

    train_size = len(namespace["load_from_disk"](str(data_dir / "dataset"))["train"])
    prepared_data.metadata["train_rows"] = train_size
    prepared_data.metadata["validation_articles"] = 500
    prepared_data.metadata["tokenizer"] = "ai-forever/rugpt3small_based_on_gpt2"
    return train_size


def run_benchmark_suite(
    source_code: str,
    prepared_data: Input[Dataset],
    benchmark_results: Output[Dataset],
    benchmark_metrics: Output[Metrics],
    benchmark_report: Output[HTML],
) -> float:
    """Run baseline, short candidates, and final on the same physical GPU"""

    import html
    import json
    import pathlib
    import subprocess
    import sys
    import tarfile

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Aviflow started the benchmark without a CUDA GPU")
    print(
        "GPU runtime:",
        f"device={torch.cuda.get_device_name(0)}",
        f"torch={torch.__version__}",
        f"cuda={torch.version.cuda}",
        flush=True,
    )

    script_path = pathlib.Path("/tmp/homework2_train.py")
    script_path.write_text(source_code, encoding="utf-8")
    results_dir = pathlib.Path("/tmp/homework2-results")
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--suite",
            "--data-dir",
            prepared_data.path,
            "--output-dir",
            str(results_dir),
        ],
        check=True,
    )

    suite = json.loads((results_dir / "suite_summary.json").read_text())
    baseline = suite["baseline"]
    final = suite["final"]
    benchmark_metrics.log_metric("baseline_tokens_per_second", baseline["tokens_per_second"])
    benchmark_metrics.log_metric("final_tokens_per_second", final["tokens_per_second"])
    benchmark_metrics.log_metric("speedup", suite["speedup"])
    benchmark_metrics.log_metric("baseline_peak_allocated_gib", baseline["peak_allocated_gib"])
    benchmark_metrics.log_metric("final_peak_allocated_gib", final["peak_allocated_gib"])
    benchmark_metrics.log_metric("baseline_validation_loss", baseline["validation_loss"])
    benchmark_metrics.log_metric("final_validation_loss", final["validation_loss"])

    artifact_path = pathlib.Path(benchmark_results.path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(artifact_path, "w:gz") as archive:
        archive.add(results_dir, arcname="homework2-results")
    benchmark_results.metadata["selected_profile"] = suite["selected_profile"]
    benchmark_results.metadata["speedup"] = suite["speedup"]
    benchmark_results.metadata["gpu"] = suite["gpu_name"]

    def format_number(value, digits=2):
        if value is None:
            return "—"
        return f"{float(value):.{digits}f}"

    rows = []
    for item in suite["experiments"]:
        config = item.get("training_config", {})
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['run_name'])}</td>"
            f"<td>{html.escape(item['profile'])}</td>"
            f"<td>{format_number(item.get('train_seconds'))}</td>"
            f"<td>{format_number(item.get('tokens_per_second'))}</td>"
            f"<td>{format_number(item.get('peak_allocated_gib'))}</td>"
            f"<td>{format_number(item.get('validation_loss'), 4)}</td>"
            f"<td>{item.get('optimizer_steps', '—')}</td>"
            f"<td>{config.get('per_device_train_batch_size', '—')} / "
            f"{config.get('gradient_accumulation_steps', '—')}</td>"
            "</tr>"
        )

    markdown_report = (results_dir / "research_report.md").read_text()
    report = f"""<!doctype html>
<html lang="ru"><meta charset="utf-8"><title>Homework 2 benchmark</title>
<style>
body{{font-family:system-ui;max-width:1200px;margin:40px auto;line-height:1.45}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:7px;text-align:right}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
pre{{white-space:pre-wrap;background:#f6f8fa;padding:16px;border-radius:8px}}
</style><body>
<h1>Qwen3 training efficiency benchmark</h1>
<p><b>GPU:</b> {html.escape(suite['gpu_name'])}<br>
<b>Selected profile:</b> {html.escape(suite['selected_profile'])}<br>
<b>Speedup S:</b> {float(suite['speedup']):.4f}x</p>
<table><thead><tr><th>Run</th><th>Profile</th><th>Seconds</th><th>Tokens/s</th>
<th>Peak GiB</th><th>Validation loss</th><th>Steps</th><th>Batch / accum</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Generated research report</h2><pre>{html.escape(markdown_report)}</pre>
</body></html>"""
    report_path = pathlib.Path(benchmark_report.path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    benchmark_report.metadata["title"] = "Homework 2 training benchmark"
    return float(suite["speedup"])


def make_pipeline(run_name: str):
    prepare_task = remote(
        runtime_env={"base_component": BASE_COMPONENT, "pip": COMMON_PACKAGES},
        cpus=16,
        memory_mb=65536,
        enable_caching=True,
    )(prepare_benchmark_data)
    benchmark_task = remote(
        runtime_env={"base_component": BASE_COMPONENT, "pip": GPU_PACKAGES},
        cpus=16,
        memory_mb=65536,
        shared_memory_mb=16384,
        gpus=1,
        enable_caching=False,
    )(run_benchmark_suite)

    @pipeline(experiment_name="llm-homework-2", run_name=run_name)
    def homework2_pipeline(source_code: str):
        prepared = prepare_task(source_code=source_code)
        benchmark_task(
            source_code=source_code,
            prepared_data=prepared.outputs["prepared_data"],
        )

    return homework2_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Homework 2 in Aviflow")
    parser.add_argument("--namespace", default="students")
    parser.add_argument("--run-name", default="qwen3-training-optimizations-v1")
    parser.add_argument("--compile", metavar="PATH")
    args = parser.parse_args()

    source_code = Path(__file__).with_name("train.py").read_text(encoding="utf-8")
    aviflow.init(args.namespace)
    homework2_pipeline = make_pipeline(args.run_name)
    parameters = {"source_code": source_code}
    if args.compile:
        homework2_pipeline.compile(path=args.compile, parameters=parameters)
    else:
        homework2_pipeline.remote(parameters=parameters)


if __name__ == "__main__":
    main()
