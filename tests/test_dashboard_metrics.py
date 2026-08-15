"""Offline coverage for dashboard metric fallbacks.

Helpers in ``app`` take injected Qdrant/Redis/inspect stand-ins, so these
tests never open a network connection or import Streamlit.
"""

import app


class _FakeQdrant:
    def __init__(self, *, exists=True, points_count=0, error=None):
        self.exists = exists
        self.points_count = points_count
        self.error = error

    def collection_exists(self, collection_name):
        if self.error:
            raise self.error
        return self.exists

    def get_collection(self, collection_name):
        if self.error:
            raise self.error

        class _Info:
            points_count = self.points_count

        return _Info()


class _FakeRedis:
    def __init__(self, *, error=None):
        self.error = error
        self.pinged = False

    def ping(self):
        self.pinged = True
        if self.error:
            raise self.error
        return True


# --------------------------------------------------------------------------- #
# metric_or_fallback
# --------------------------------------------------------------------------- #
def test_fallback_error_is_not_connected():
    assert app.metric_or_fallback(error=True) == app.NOT_CONNECTED


def test_fallback_missing_or_none_is_zero():
    assert app.metric_or_fallback(missing=True) == 0
    assert app.metric_or_fallback(value=None) == 0


def test_fallback_returns_integer_count():
    assert app.metric_or_fallback(value=1200) == 1200


# --------------------------------------------------------------------------- #
# fetch_document_count
# --------------------------------------------------------------------------- #
def test_document_count_missing_collection_is_zero():
    assert app.fetch_document_count(_FakeQdrant(exists=False), "streamrag_entries") == 0


def test_document_count_connection_error_is_not_connected():
    client = _FakeQdrant(error=ConnectionError("qdrant is down"))
    assert app.fetch_document_count(client) == app.NOT_CONNECTED


def test_document_count_uses_points_count():
    assert app.fetch_document_count(_FakeQdrant(exists=True, points_count=7)) == 7


def test_document_count_none_points_count_is_zero():
    assert app.fetch_document_count(_FakeQdrant(exists=True, points_count=None)) == 0


# --------------------------------------------------------------------------- #
# fetch_active_streams
# --------------------------------------------------------------------------- #
def test_active_streams_redis_down_is_not_connected():
    redis_client = _FakeRedis(error=ConnectionError("redis is down"))
    called = []
    value = app.fetch_active_streams(redis_client, lambda: called.append(True) or {})
    assert value == app.NOT_CONNECTED
    assert called == []


def test_active_streams_no_workers_is_zero():
    assert app.fetch_active_streams(_FakeRedis(), lambda: None) == 0


def test_active_streams_counts_workers():
    def inspect():
        return {"celery@w1": [], "celery@w2": [{"id": "t1"}]}

    assert app.fetch_active_streams(_FakeRedis(), inspect) == 2


def test_active_streams_inspect_error_after_ping_is_zero():
    def boom():
        raise TimeoutError("inspect timed out")

    assert app.fetch_active_streams(_FakeRedis(), boom) == 0
