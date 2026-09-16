# Single shared image for every BioForge microservice - which one a
# container actually runs is selected at `docker run`/compose/k8s time via
# the SERVICE env var (see docker/entrypoint.sh). This keeps the image build
# graph simple (one image, one dependency set) while every service still
# runs as its own independently-scaled container/pod - the modularity lives
# at the deployment layer (docker-compose.yml / k8s/), not the image layer.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY engines/pharmaforge_core/pyproject.toml engines/pharmaforge_core/pyproject.toml
COPY engines/pharmaforge_core/pharmaforge_core engines/pharmaforge_core/pharmaforge_core
COPY engines/pharmaforge_core/data_cache engines/pharmaforge_core/data_cache
COPY engines/pharmaforge_core/checkpoints engines/pharmaforge_core/checkpoints
COPY pyproject.toml pyproject.toml
COPY src src

RUN pip install --no-cache-dir -e ./engines/pharmaforge_core \
    && pip install --no-cache-dir -e ".[scale,tracking,physics]"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000-8008
ENTRYPOINT ["/entrypoint.sh"]
