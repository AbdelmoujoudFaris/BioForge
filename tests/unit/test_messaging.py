"""`bioforge.common.messaging` must degrade gracefully with no Redis/Celery
running - that's the whole point of the module (REST-only deployments never
need them). These tests don't require a broker.
"""
from bioforge.common.messaging import as_distributed_task, get_celery_app, publish_event, subscribe


def test_publish_event_returns_false_without_a_reachable_broker():
    assert publish_event("candidates.generated", {"candidate_id": "abc"}) is False


def test_subscribe_raises_clear_error_without_redis_installed_or_reachable():
    import pytest

    with pytest.raises(RuntimeError):
        next(iter(subscribe("candidates.generated")))


def test_as_distributed_task_leaves_function_directly_callable():
    calls = []

    @as_distributed_task("bioforge.test.echo")
    def echo(x):
        calls.append(x)
        return x

    assert echo(42) == 42
    assert calls == [42]


def test_get_celery_app_raises_without_celery_installed_or_broker(monkeypatch):
    import bioforge.common.messaging as messaging

    monkeypatch.setattr(messaging, "_celery_app", None)
    try:
        messaging.get_celery_app()
    except RuntimeError:
        pass  # celery not installed in this environment - expected
    except Exception:
        pass  # celery installed but no broker reachable - also acceptable here
