FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel AS base

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip \
    && pip install 'transformers==4.52.0' 'datasets==3.6.0' 'wandb==0.19.10' 'accelerate==1.6.0' 'matplotlib==3.10.1'

RUN python3 -m pip install --no-cache-dir --no-build-isolation \
    'flash-attn==2.7.3'

ENV HF_DISABLE_TELEMETRY=true \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    SERVICE_HOST=0.0.0.0 \
    SERVICE_PORT=8890 \
    PROJECT_ROOT=/app

ENV PATH="$PROJECT_ROOT/bin:$PATH" PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

COPY your_solution.py $PROJECT_ROOT/your_solution.py

CMD ["python3", "your_solution.py", "all"]
