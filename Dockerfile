# ==========================
# 1️⃣ Builder stage
# ==========================
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --timeout 300 --retries 10 -r requirements.txt

# ==========================
# 2️⃣ Runtime stage
# ==========================
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG UID=1000
ARG GID=1000

RUN groupadd -g ${GID} appgroup && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ENV HOME=/home/appuser \
    USER=appuser \
    TORCH_HOME=/home/appuser/.cache/torch \
    TORCHINDUCTOR_CACHE_DIR=/home/appuser/.cache/torchinductor \
    HF_HOME=/data/hf \
    HF_HUB_CACHE=/data/hf/hub \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /workspace/segmentation

COPY app/ ./

RUN mkdir -p /data/hf /home/appuser/.cache/torch /home/appuser/.cache/torchinductor && \
    chown -R appuser:appgroup /workspace/segmentation /data/hf /home/appuser

USER appuser

CMD ["python3", "cli.py"]