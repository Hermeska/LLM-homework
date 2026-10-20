"""Execute Homework 3 end-to-end on one Aviflow GPU."""

import argparse
import base64
import importlib
import io
import tarfile
from pathlib import Path

import aviflow
from aviflow import Dataset, HTML, Metrics, Output, pipeline, remote


_aviflow_remote_module = importlib.import_module("aviflow.remote")
_aviflow_remote_module.PYPI_INDEX_URLS = [
    "https://download.pytorch.org/whl/cu128",
    "https://pypi.org/simple",
]

BASE_COMPONENT = "pytorch-dl/pytorch-train-dl:0.2.4"
RUNTIME_PACKAGES = [
    "transformers==4.56.1",
    "trl==0.23.1",
    "datasets==3.6.0",
    "accelerate==1.6.0",
    "ipywidgets==8.1.9",
    "matplotlib==3.10.1",
    "pandas==2.2.3",
    "nbclient==0.10.2",
    "nbconvert==7.16.6",
    "nbformat==5.10.4",
    "ipykernel==6.30.1",
]


def execute_homework(
    input_bundle_b64: str,
    completed_homework: Output[Dataset],
    evaluation_metrics: Output[Metrics],
    notebook_report: Output[HTML],
) -> float:
    """Train, evaluate, and return the executed notebook without model weights."""

    import base64
    import io
    import json
    import os
    import pathlib
    import shutil
    import tarfile

    import nbformat
    import pandas as pd
    import torch
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    if not torch.cuda.is_available():
        raise RuntimeError("Aviflow started HW3 without a CUDA GPU")
    print(
        "GPU runtime:",
        f"device={torch.cuda.get_device_name(0)}",
        f"torch={torch.__version__}",
        f"cuda={torch.version.cuda}",
    )

    work_dir = pathlib.Path("/tmp/hw3")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    bundle = base64.b64decode(input_bundle_b64)
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:xz") as archive:
        archive.extractall(work_dir, filter="data")

    source_path = work_dir / "sft_function_calling_hw_student.ipynb"
    output_path = work_dir / "sft_function_calling_hw.ipynb"
    notebook = nbformat.read(source_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(work_dir)}},
    )
    previous_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        client.execute()
    finally:
        os.chdir(previous_cwd)
    nbformat.write(notebook, output_path)

    csv_path = work_dir / "outputs" / "eval_results.csv"
    if not csv_path.exists():
        raise RuntimeError("Notebook finished without outputs/eval_results.csv")
    scores = pd.read_csv(csv_path)
    print("Evaluation results:\n", scores.to_string(index=False))
    for row in scores.to_dict(orient="records"):
        prefix = f"{row['model']}_{row['lang']}"
        evaluation_metrics.log_metric(
            f"{prefix}_tool_set_accuracy", float(row["tool_set_accuracy"])
        )
        evaluation_metrics.log_metric(
            f"{prefix}_exact_match", float(row["exact_match"])
        )

    test_rows = [
        json.loads(line)
        for line in (work_dir / "data" / "test_cases.jsonl").read_text().splitlines()
        if line.strip()
    ]
    lang_counts = pd.Series([row["lang"] for row in test_rows]).value_counts()
    sft_rows = scores[scores["model"] == "sft"].set_index("lang")
    total = float(lang_counts.sum())
    sft_exact_match = sum(
        float(sft_rows.loc[lang, "exact_match"]) * int(count)
        for lang, count in lang_counts.items()
    ) / total
    evaluation_metrics.log_metric("sft_exact_match_overall", sft_exact_match)

    artifact_path = pathlib.Path(completed_homework.path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(artifact_path, "w:gz") as archive:
        archive.add(output_path, arcname="sft_function_calling_hw.ipynb")
        archive.add(csv_path, arcname="outputs/eval_results.csv")
    completed_homework.metadata["gpu"] = torch.cuda.get_device_name(0)
    completed_homework.metadata["contains_model_weights"] = False
    completed_homework.metadata["sft_exact_match_overall"] = sft_exact_match

    html_exporter = HTMLExporter()
    html_exporter.exclude_input_prompt = True
    html_exporter.exclude_output_prompt = True
    html_body, _ = html_exporter.from_notebook_node(notebook)
    report_path = pathlib.Path(notebook_report.path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html_body, encoding="utf-8")
    notebook_report.metadata["title"] = "HW3: SFT for function calling"
    return sft_exact_match


def make_pipeline(run_name: str):
    execute_task = remote(
        runtime_env={"base_component": BASE_COMPONENT, "pip": RUNTIME_PACKAGES},
        cpus=16,
        memory_mb=65536,
        shared_memory_mb=16384,
        gpus=1,
        enable_caching=False,
    )(execute_homework)

    @pipeline(experiment_name="llm-homework-3", run_name=run_name)
    def homework_pipeline(input_bundle_b64: str):
        execute_task(input_bundle_b64=input_bundle_b64)

    return homework_pipeline


def build_input_bundle(homework_dir: Path) -> str:
    buffer = io.BytesIO()
    names = [
        "sft_function_calling_hw_student.ipynb",
        "smart_home.py",
        "chat_ui.py",
        "data/train.jsonl",
        "data/validation.jsonl",
        "data/test_cases.jsonl",
    ]
    # KFP passes parameters to the executor through a command-line argument.
    # XZ keeps this bundle below Linux's per-argument size limit; gzip does not.
    with tarfile.open(fileobj=buffer, mode="w:xz") as archive:
        for name in names:
            archive.add(homework_dir / name, arcname=name)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description="Launch HW3 SFT in Aviflow")
    parser.add_argument("--namespace", default="students")
    parser.add_argument("--run-name", default="qwen25-sft-function-calling-v1")
    parser.add_argument("--compile", metavar="PATH")
    args = parser.parse_args()

    homework_dir = Path(__file__).parent
    input_bundle_b64 = build_input_bundle(homework_dir)
    if len(input_bundle_b64.encode("ascii")) >= 120_000:
        raise ValueError(
            "The encoded homework bundle is too large for a KFP parameter: "
            f"{len(input_bundle_b64.encode('ascii'))} bytes"
        )
    parameters = {"input_bundle_b64": input_bundle_b64}
    aviflow.init(args.namespace)
    homework_pipeline = make_pipeline(args.run_name)
    if args.compile:
        homework_pipeline.compile(path=args.compile, parameters=parameters)
    else:
        homework_pipeline.remote(parameters=parameters)


if __name__ == "__main__":
    main()
