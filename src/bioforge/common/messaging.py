"""Inter-service messaging and distributed-compute scaffolding.

Every BioForge service is reachable over plain REST (see each
`services/*/main.py`), which is what the orchestrator graph and the SDK use
by default and all that's required to run the whole platform on a laptop.
This module adds the optional scale-out path described in the platform
requirements:

  * `publish_event` / `subscribe` - Redis pub/sub for service-to-service
    eventing (e.g. "candidate generated" fanning out to refine/retrieval/
    admet workers) when `pip install bioforge[scale]` is present.
  * `celery_app` - a lazily-constructed Celery application, so a service can
    submit a stage as a distributed task (`generate_candidates.delay(...)`)
    for GPU-scheduled worker pools instead of running it inline.

Both are optional: every service function is also directly callable
in-process (that's what the test suite and `orchestrator/graph.py` use by
default), so nobody needs a running Redis/Celery cluster to develop or test
BioForge locally.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Iterator

from bioforge.common.config import get_settings

logger = logging.getLogger(__name__)


def _try_import_redis():
    try:
        import redis  # type: ignore

        return redis
    except ImportError:
        return None


def publish_event(channel: str, payload: dict[str, Any]) -> bool:
    """Publish a JSON event. Returns False (and logs) if Redis isn't
    installed/reachable rather than raising, since eventing is an
    optimization, not required for the request/response REST path.
    """
    redis_mod = _try_import_redis()
    if redis_mod is None:
        logger.debug("redis-py not installed; skipping publish_event(%s)", channel)
        return False
    try:
        client = redis_mod.from_url(get_settings().redis_url)
        client.publish(channel, json.dumps(payload, default=str))
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Redis publish failed for channel %s: %s", channel, exc)
        return False


def subscribe(channel: str) -> Iterator[dict[str, Any]]:
    redis_mod = _try_import_redis()
    if redis_mod is None:
        raise RuntimeError("redis-py not installed; `pip install bioforge[scale]` to use subscribe()")
    client = redis_mod.from_url(get_settings().redis_url)
    pubsub = client.pubsub()
    pubsub.subscribe(channel)
    for message in pubsub.listen():
        if message["type"] != "message":
            continue
        yield json.loads(message["data"])


_celery_app = None


def get_celery_app():
    """Lazily construct the shared Celery app. Import-time construction is
    avoided so services that never touch distributed compute (most unit
    tests, the gateway's simple proxy routes) don't need celery installed.
    """
    global _celery_app
    if _celery_app is not None:
        return _celery_app
    try:
        from celery import Celery  # type: ignore
    except ImportError as exc:
        raise RuntimeError("celery not installed; `pip install bioforge[scale]` to use get_celery_app()") from exc

    settings = get_settings()
    _celery_app = Celery("bioforge", broker=settings.celery_broker_url, backend=settings.celery_broker_url)
    _celery_app.conf.update(task_serializer="json", result_serializer="json", accept_content=["json"])
    return _celery_app


def as_distributed_task(name: str) -> Callable[[Callable], Callable]:
    """Decorator registering `fn` as a Celery task under `name` *if* Celery
    is installed, while leaving `fn` directly callable either way. Services
    call `fn(...)` for in-process/dev execution and `fn.delay(...)` (only
    available when celery is installed) for GPU-worker-pool execution.
    """

    def decorator(fn: Callable) -> Callable:
        try:
            app = get_celery_app()
        except RuntimeError:
            return fn
        return app.task(name=name)(fn)

    return decorator
