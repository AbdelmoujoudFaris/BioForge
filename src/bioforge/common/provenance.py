"""Run provenance tracking.

Uses MLflow when it's installed and `BIOFORGE_MLFLOW_TRACKING_URI` (or a
local `mlruns/` default) is reachable; otherwise falls back to an
append-only JSON-lines log under `.provenance/`, so every pipeline run is
still reproducible (params + versions + outputs recorded) without forcing
every contributor to run an MLflow server. Never silently drops a run: if
neither backend can write, the exception propagates.
"""
from __future__ import annotations

import json
import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

from bioforge.common.config import get_settings

logger = logging.getLogger(__name__)


def _try_import_mlflow():
    try:
        import mlflow  # type: ignore

        return mlflow
    except ImportError:
        return None


@dataclass
class RunHandle:
    run_id: str
    backend: str  # "mlflow" | "jsonl"
    _params: dict[str, Any] = field(default_factory=dict)
    _metrics: dict[str, float] = field(default_factory=dict)
    _artifacts: dict[str, Any] = field(default_factory=dict)
    _mlflow_active_run: Any = None

    def log_params(self, **params: Any) -> None:
        self._params.update(params)
        mlflow = _try_import_mlflow()
        if self.backend == "mlflow" and mlflow is not None:
            mlflow.log_params(params)

    def log_metrics(self, **metrics: float) -> None:
        self._metrics.update(metrics)
        mlflow = _try_import_mlflow()
        if self.backend == "mlflow" and mlflow is not None:
            mlflow.log_metrics(metrics)

    def log_artifact_ref(self, name: str, value: Any) -> None:
        """Record a lightweight, JSON-serializable pointer to an artifact
        (a file path, a version string, a dataset hash) - not the artifact
        bytes themselves, to keep provenance logs small.
        """
        self._artifacts[name] = value


@contextmanager
def track_run(stage: str, engine_versions: dict[str, str] | None = None) -> Iterator[RunHandle]:
    """Context manager wrapping one pipeline stage's provenance record.

    `engine_versions` should capture the versions of every model/adapter
    actually invoked (e.g. {"pharmaforge_core": "0.1.0", "generator": "equivariant_diffusion"})
    so a later reader can tell exactly what produced a given output.
    """
    settings = get_settings()
    run_id = str(uuid.uuid4())
    mlflow = _try_import_mlflow()
    backend = "jsonl"
    active_run = None

    if mlflow is not None:
        try:
            if settings.mlflow_tracking_uri:
                mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment("bioforge")
            active_run = mlflow.start_run(run_name=f"{stage}-{run_id[:8]}")
            backend = "mlflow"
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("MLflow unavailable (%s); falling back to local JSONL provenance log", exc)
            backend = "jsonl"

    handle = RunHandle(run_id=run_id, backend=backend, _mlflow_active_run=active_run)
    handle.log_params(stage=stage, **(engine_versions or {}))
    error: Exception | None = None
    try:
        yield handle
    except Exception as exc:
        error = exc
        raise
    finally:
        record = {
            "run_id": run_id,
            "stage": stage,
            "backend": backend,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params": handle._params,
            "metrics": handle._metrics,
            "artifacts": handle._artifacts,
            "error": str(error) if error else None,
        }
        if backend == "mlflow" and mlflow is not None and active_run is not None:
            for name, value in handle._artifacts.items():
                mlflow.set_tag(f"artifact.{name}", str(value))
            mlflow.end_run(status="FAILED" if error else "FINISHED")
        # Always also write the JSONL trail, even on the mlflow path, so
        # provenance survives without an MLflow server to query later.
        settings.provenance_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = settings.provenance_log_dir / f"{stage}.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
