"""Offline coverage for worker.py's embedding/Qdrant-indexing step.

Every test replaces ``worker.qdrant`` and ``worker._get_embeddings`` with
fakes, so no live network call, real Qdrant instance, or OpenAI API key is
ever required.
"""

from datetime import datetime, timezone
from email.utils import format_datetime

import pytest

import scoring
import worker


class FakeEmbeddings:
    """Records the texts it was asked to embed and returns fixed-size vectors."""

    def __init__(self, dimension: int = 3):
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        # Deterministic per-text vector (its length lets tests spot which
        # text produced which vector without depending on real embeddings).
        return [[float(len(text)), 0.0, 1.0] for text in texts]


class FakeQdrantClient:
    """Records collection/upsert calls; ``exists`` controls collection_exists()."""

    def __init__(self, exists: bool = False):
        self.exists = exists
        self.created: list[dict] = []
        self.upserted_points: list = []
        self.upsert_calls: list[dict] = []

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def create_collection(self, collection_name: str, vectors_config) -> None:
        self.created.append({"collection_name": collection_name, "vectors_config": vectors_config})
        self.exists = True

    def upsert(self, collection_name: str, points) -> None:
        self.upsert_calls.append({"collection_name": collection_name, "points": points})
        self.upserted_points.extend(points)


def _entry(title="Title", link="https://example.com/a", published="", summary="Summary"):
    return {"title": title, "link": link, "published": published, "summary": summary}


# --------------------------------------------------------------------------- #
# _point_id_for_entry
# --------------------------------------------------------------------------- #
def test_point_id_is_deterministic_for_same_link():
    entry_a = _entry(link="https://example.com/same")
    entry_b = _entry(title="Different Title", link="https://example.com/same")
    assert worker._point_id_for_entry(entry_a) == worker._point_id_for_entry(entry_b)


def test_point_id_differs_for_different_links():
    id_a = worker._point_id_for_entry(_entry(link="https://example.com/a"))
    id_b = worker._point_id_for_entry(_entry(link="https://example.com/b"))
    assert id_a != id_b


def test_point_id_falls_back_to_title_when_link_missing():
    entry_a = _entry(title="Only Title", link="")
    entry_b = _entry(title="Only Title", link="")
    entry_c = _entry(title="Other Title", link="")
    assert worker._point_id_for_entry(entry_a) == worker._point_id_for_entry(entry_b)
    assert worker._point_id_for_entry(entry_a) != worker._point_id_for_entry(entry_c)


# --------------------------------------------------------------------------- #
# _ensure_collection
# --------------------------------------------------------------------------- #
def test_ensure_collection_creates_when_missing(monkeypatch):
    fake = FakeQdrantClient(exists=False)
    monkeypatch.setattr(worker, "qdrant", fake)

    worker._ensure_collection(3)

    assert len(fake.created) == 1
    assert fake.created[0]["collection_name"] == worker.QDRANT_COLLECTION
    vectors_config = fake.created[0]["vectors_config"]
    assert vectors_config.size == 3
    from qdrant_client.models import Distance

    assert vectors_config.distance == Distance.COSINE


def test_ensure_collection_skips_when_already_exists(monkeypatch):
    fake = FakeQdrantClient(exists=True)
    monkeypatch.setattr(worker, "qdrant", fake)

    worker._ensure_collection(3)

    assert fake.created == []


# --------------------------------------------------------------------------- #
# _embed_and_index
# --------------------------------------------------------------------------- #
def test_embed_and_index_empty_entries_is_a_noop(monkeypatch):
    fake_client = FakeQdrantClient()
    fake_embeddings = FakeEmbeddings()
    monkeypatch.setattr(worker, "qdrant", fake_client)
    monkeypatch.setattr(worker, "_get_embeddings", lambda: fake_embeddings)

    assert worker._embed_and_index([]) == 0
    assert fake_embeddings.calls == []
    assert fake_client.upsert_calls == []
    assert fake_client.created == []


def test_embed_and_index_upserts_one_point_per_entry_with_expected_payload(monkeypatch):
    fake_client = FakeQdrantClient(exists=False)
    fake_embeddings = FakeEmbeddings()
    monkeypatch.setattr(worker, "qdrant", fake_client)
    monkeypatch.setattr(worker, "_get_embeddings", lambda: fake_embeddings)

    entries = [
        _entry(title="First", link="https://example.com/1", published="d1", summary="s1"),
        _entry(title="Second", link="https://example.com/2", published="d2", summary="s2"),
    ]

    count = worker._embed_and_index(entries)

    assert count == 2
    assert len(fake_client.upserted_points) == 2

    point_by_link = {p.payload["link"]: p for p in fake_client.upserted_points}
    first = point_by_link["https://example.com/1"]
    assert first.payload["title"] == "First"
    assert first.payload["link"] == "https://example.com/1"
    assert first.payload["published"] == "d1"
    assert first.payload["summary"] == "s1"
    # "d1" is not a parseable RSS date, so the stored weight is the unknown default.
    assert first.payload["recency_weight"] == scoring.UNKNOWN_DATE_WEIGHT
    assert first.id == worker._point_id_for_entry(entries[0])

    # The collection was created (it didn't exist) before the upsert.
    assert len(fake_client.created) == 1


def test_embed_and_index_reuses_existing_collection(monkeypatch):
    fake_client = FakeQdrantClient(exists=True)
    fake_embeddings = FakeEmbeddings()
    monkeypatch.setattr(worker, "qdrant", fake_client)
    monkeypatch.setattr(worker, "_get_embeddings", lambda: fake_embeddings)

    worker._embed_and_index([_entry()])

    assert fake_client.created == []


def test_embed_and_index_repolling_same_link_upserts_not_duplicates(monkeypatch):
    fake_client = FakeQdrantClient(exists=False)
    fake_embeddings = FakeEmbeddings()
    monkeypatch.setattr(worker, "qdrant", fake_client)
    monkeypatch.setattr(worker, "_get_embeddings", lambda: fake_embeddings)

    entry = _entry(link="https://example.com/repeat")
    worker._embed_and_index([entry])
    worker._embed_and_index([entry])

    # Two separate upsert calls, but both point at the same deterministic ID
    # (a real Qdrant server would treat this as an update, not a duplicate).
    assert len(fake_client.upsert_calls) == 2
    ids = {p.id for p in fake_client.upserted_points}
    assert ids == {worker._point_id_for_entry(entry)}


def test_embed_and_index_stores_near_one_recency_for_just_published(monkeypatch):
    fake_client = FakeQdrantClient(exists=True)
    fake_embeddings = FakeEmbeddings()
    monkeypatch.setattr(worker, "qdrant", fake_client)
    monkeypatch.setattr(worker, "_get_embeddings", lambda: fake_embeddings)

    now = datetime(2021, 9, 7, 12, 0, 0, tzinfo=timezone.utc)
    worker._embed_and_index(
        [_entry(published=format_datetime(now))],
        now=now,
    )

    weight = fake_client.upserted_points[0].payload["recency_weight"]
    assert weight == pytest.approx(1.0)


def test_embed_and_index_stores_zero_recency_when_published_missing(monkeypatch):
    fake_client = FakeQdrantClient(exists=True)
    fake_embeddings = FakeEmbeddings()
    monkeypatch.setattr(worker, "qdrant", fake_client)
    monkeypatch.setattr(worker, "_get_embeddings", lambda: fake_embeddings)

    worker._embed_and_index([_entry(published="")], now=datetime.now(timezone.utc))

    payload = fake_client.upserted_points[0].payload
    assert payload["recency_weight"] == scoring.UNKNOWN_DATE_WEIGHT
    assert payload["title"] == "Title"
    assert payload["link"] == "https://example.com/a"
    assert payload["published"] == ""
    assert payload["summary"] == "Summary"


def test_embed_and_index_propagates_embedding_failure(monkeypatch):
    class BoomEmbeddings:
        def embed_documents(self, texts):
            raise RuntimeError("embedding API is down")

    fake_client = FakeQdrantClient()
    monkeypatch.setattr(worker, "qdrant", fake_client)
    monkeypatch.setattr(worker, "_get_embeddings", lambda: BoomEmbeddings())

    try:
        worker._embed_and_index([_entry()])
        assert False, "expected RuntimeError to propagate"
    except RuntimeError as exc:
        assert "embedding API is down" in str(exc)
    assert fake_client.upsert_calls == []


def test_embed_and_index_propagates_upsert_failure(monkeypatch):
    class BoomQdrantClient(FakeQdrantClient):
        def upsert(self, collection_name, points):
            raise RuntimeError("qdrant upsert failed")

    fake_embeddings = FakeEmbeddings()
    monkeypatch.setattr(worker, "qdrant", BoomQdrantClient(exists=True))
    monkeypatch.setattr(worker, "_get_embeddings", lambda: fake_embeddings)

    try:
        worker._embed_and_index([_entry()])
        assert False, "expected RuntimeError to propagate"
    except RuntimeError as exc:
        assert "qdrant upsert failed" in str(exc)


# --------------------------------------------------------------------------- #
# _get_embeddings
# --------------------------------------------------------------------------- #
def test_get_embeddings_is_cached_across_calls(monkeypatch):
    monkeypatch.setattr(worker, "_embeddings", None)
    created = []

    class DummyEmbeddings:
        def __init__(self, model):
            self.model = model
            created.append(self)

    monkeypatch.setattr(worker, "OpenAIEmbeddings", DummyEmbeddings)

    first = worker._get_embeddings()
    second = worker._get_embeddings()

    assert first is second
    assert len(created) == 1


# --------------------------------------------------------------------------- #
# process_rss_feed wiring
# --------------------------------------------------------------------------- #
def test_process_rss_feed_returns_embed_and_index_result(monkeypatch):
    monkeypatch.setattr(worker, "_assert_public_host", lambda feed_url: None)
    monkeypatch.setattr(worker, "_parse_feed_with_timeout", lambda feed_url: _FakeFeed())
    monkeypatch.setattr(worker, "_embed_and_index", lambda entries: 42)

    result = worker.process_rss_feed("https://example.com/feed.xml")

    assert result == 42


def test_process_rss_feed_passes_deduped_entries_to_embed_and_index(monkeypatch):
    monkeypatch.setattr(worker, "_assert_public_host", lambda feed_url: None)
    monkeypatch.setattr(worker, "_parse_feed_with_timeout", lambda feed_url: _FakeFeed())

    received = {}

    def fake_embed_and_index(entries):
        received["entries"] = entries
        return len(entries)

    monkeypatch.setattr(worker, "_embed_and_index", fake_embed_and_index)

    worker.process_rss_feed("https://example.com/feed.xml")

    assert [e["link"] for e in received["entries"]] == ["https://example.com/only"]


def test_process_rss_feed_propagates_embed_and_index_failure(monkeypatch):
    monkeypatch.setattr(worker, "_assert_public_host", lambda feed_url: None)
    monkeypatch.setattr(worker, "_parse_feed_with_timeout", lambda feed_url: _FakeFeed())

    def boom(entries):
        raise RuntimeError("indexing exploded")

    monkeypatch.setattr(worker, "_embed_and_index", boom)

    try:
        worker.process_rss_feed("https://example.com/feed.xml")
        assert False, "expected RuntimeError to propagate"
    except RuntimeError as exc:
        assert "indexing exploded" in str(exc)


class _FakeFeed:
    """Minimal feedparser-shaped object with two entries sharing one link."""

    bozo = False
    entries = [
        {"title": "Dup", "link": "https://example.com/only", "published": "p", "summary": "s"},
        {"title": "Dup again", "link": "https://example.com/only", "published": "p", "summary": "s"},
    ]
