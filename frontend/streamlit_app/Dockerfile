FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY engines/pharmaforge_core/pyproject.toml engines/pharmaforge_core/pyproject.toml
COPY engines/pharmaforge_core/pharmaforge_core engines/pharmaforge_core/pharmaforge_core
COPY engines/pharmaforge_core/data_cache engines/pharmaforge_core/data_cache
COPY engines/pharmaforge_core/checkpoints engines/pharmaforge_core/checkpoints
COPY pyproject.toml pyproject.toml
COPY src src
COPY frontend frontend

RUN pip install --no-cache-dir -e ./engines/pharmaforge_core \
    && pip install --no-cache-dir -e ".[frontend]"

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "frontend/streamlit_app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
