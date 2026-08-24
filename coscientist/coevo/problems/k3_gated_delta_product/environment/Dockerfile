FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCHINDUCTOR_CACHE_DIR=/home/agent/.cache/torchinductor \
    TRITON_CACHE_DIR=/home/agent/.cache/triton

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    git \
    ninja-build \
    tmux \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 --shell /bin/bash agent

WORKDIR /app
COPY reference_model.py candidate.py scorer.py evaluate.py /app/

RUN mkdir -p /orig /logs/verifier /home/agent/.cache && \
    cp /app/reference_model.py /orig/reference_model.py && \
    cp /app/scorer.py /orig/scorer.py && \
    sha256sum /app/reference_model.py /app/scorer.py /app/evaluate.py > /orig/protected.sha256 && \
    chown -R agent:agent /app /home/agent && \
    chmod 700 /orig && \
    chmod 755 /app/evaluate.py /app/scorer.py

CMD ["bash"]
