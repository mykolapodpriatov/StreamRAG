"""Offline coverage for the query side.

Nothing here touches a real Qdrant server or an embedding API: an in-memory
Qdrant client holds the collection and a fake embedder produces deterministic
vectors, mirroring how ``test_worker_qdrant.py`` keeps the ingestion side
offline.
"""

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import config
import scoring
from retrieval import EmbeddingModelMismatchError, RetrievedPassage, blend_score, retrieve

COLLECTION = "test_entries"
NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


class FakeEmbedder:
    """Maps a handful of known texts to fixed 2-D unit vectors.

    Cosine similarity between these is exact and hand-checkable, so a ranking
    assertion is about the ranking rather than about an embedding model.
    """

    VECTORS = {
        "rocket": [1.0, 0.0],
        "spaceflight": [0.92, 0.39],  # close to "rocket"
        "gardening": [0.0, 1.0],  # orthogonal
    }

    def __init__(self):
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self.VECTORS[text]


def _rfc2822(dt: datetime) -> str:
    return format_datetime(dt)


def _client_with(points: list[PointStruct]) -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    if points:
        client.upsert(collection_name=COLLECTION, points=points)
    return client


def _point(
    point_id: int,
    vector: list[float],
    *,
    title: str,
    published: datetime | None,
    model: str | None = config.EMBEDDING_MODEL,
) -> PointStruct:
    payload = {
        "title": title,
        "link": f"https://example.com/{title}",
        "published": _rfc2822(published) if published else "",
        "summary": f"{title} summary",
    }
    if model is not None:
        payload["embedding_model"] = model
    return PointStruct(id=point_id, vector=vector, payload=payload)


# --------------------------------------------------------------------------- #
# blend_score
# --------------------------------------------------------------------------- #


def test_blend_is_pure_similarity_at_zero_weight():
    assert blend_score(0.8, 0.1, 0.0) == pytest.approx(0.8)


def test_blend_is_pure_recency_at_full_weight():
    assert blend_score(0.8, 0.1, 1.0) == pytest.approx(0.1)


def test_blend_is_a_convex_combination():
    assert blend_score(1.0, 0.0, 0.25) == pytest.approx(0.75)
    assert blend_score(0.0, 1.0, 0.25) == pytest.approx(0.25)


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_blend_rejects_a_weight_outside_the_unit_interval(weight):
    with pytest.raises(ValueError, match="recency weight"):
        blend_score(0.5, 0.5, weight)


# --------------------------------------------------------------------------- #
# retrieve: ranking
# --------------------------------------------------------------------------- #


def test_similarity_alone_ranks_the_closer_match_first():
    client = _client_with(
        [
            _point(1, [1.0, 0.0], title="close", published=NOW - timedelta(days=400)),
            _point(2, [0.0, 1.0], title="far", published=NOW),
        ]
    )

    hits = retrieve("rocket", k=2, client=client, collection=COLLECTION,
                    embedder=FakeEmbedder(), weight=0.0, now=NOW)

    assert [h.title for h in hits] == ["close", "far"]


def test_recency_can_flip_the_order():
    """The property the blend exists for: an old perfect match loses to a good
    recent one once recency carries enough of the score."""
    points = [
        _point(1, [1.0, 0.0], title="old-perfect", published=NOW - timedelta(days=400)),
        _point(2, [0.92, 0.39], title="new-good", published=NOW),
    ]

    by_similarity = retrieve("rocket", k=2, client=_client_with(points), collection=COLLECTION,
                             embedder=FakeEmbedder(), weight=0.0, now=NOW)
    by_blend = retrieve("rocket", k=2, client=_client_with(points), collection=COLLECTION,
                        embedder=FakeEmbedder(), weight=0.6, now=NOW)

    assert [h.title for h in by_similarity] == ["old-perfect", "new-good"]
    assert [h.title for h in by_blend] == ["new-good", "old-perfect"]


def test_recency_is_recomputed_at_query_time_not_read_from_the_payload():
    """The stored recency_weight was true when the entry was indexed and decays
    out of date from that moment."""
    published = NOW - timedelta(days=200)
    point = _point(1, [1.0, 0.0], title="stale", published=published)
    point.payload["recency_weight"] = 1.0  # as if indexed the day it was published

    hits = retrieve("rocket", k=1, client=_client_with([point]), collection=COLLECTION,
                    embedder=FakeEmbedder(), weight=0.5, now=NOW)

    assert hits[0].recency == pytest.approx(scoring.recency_weight(_rfc2822(published), now=NOW))
    assert hits[0].recency < 1.0


def test_an_undated_entry_is_ranked_not_dropped():
    """No date is a gap in the feed's metadata, not evidence the entry is old."""
    client = _client_with(
        [
            _point(1, [1.0, 0.0], title="undated", published=None),
            _point(2, [0.0, 1.0], title="dated-but-irrelevant", published=NOW),
        ]
    )

    hits = retrieve("rocket", k=2, client=client, collection=COLLECTION,
                    embedder=FakeEmbedder(), weight=0.3, now=NOW)

    titles = [h.title for h in hits]
    assert "undated" in titles
    undated = next(h for h in hits if h.title == "undated")
    assert undated.recency == scoring.UNKNOWN_DATE_WEIGHT
    # A strong undated match still beats a weak recent one at this weight.
    assert titles[0] == "undated"


def test_results_carry_the_link_and_the_date():
    client = _client_with([_point(1, [1.0, 0.0], title="story", published=NOW)])

    hit = retrieve("rocket", k=1, client=client, collection=COLLECTION,
                   embedder=FakeEmbedder(), now=NOW)[0]

    assert isinstance(hit, RetrievedPassage)
    assert hit.link == "https://example.com/story"
    assert hit.published == _rfc2822(NOW)
    assert hit.summary == "story summary"


def test_the_candidate_pool_is_wider_than_k():
    """Recency can promote an entry outside the top-k by similarity, so it has
    to be fetched in the first place."""
    points = [
        _point(1, [1.0, 0.0], title="a-old", published=NOW - timedelta(days=500)),
        _point(2, [1.0, 0.0], title="b-old", published=NOW - timedelta(days=500)),
        _point(3, [0.92, 0.39], title="c-new", published=NOW),
    ]

    hits = retrieve("rocket", k=1, client=_client_with(points), collection=COLLECTION,
                    embedder=FakeEmbedder(), weight=0.8, now=NOW, candidate_multiplier=4)

    assert [h.title for h in hits] == ["c-new"]


def test_ranking_is_stable_across_runs():
    points = [
        _point(1, [1.0, 0.0], title="a", published=NOW),
        _point(2, [1.0, 0.0], title="b", published=NOW),
    ]
    client = _client_with(points)
    kwargs = dict(client=client, collection=COLLECTION, embedder=FakeEmbedder(), now=NOW)

    first = [h.title for h in retrieve("rocket", k=2, **kwargs)]
    second = [h.title for h in retrieve("rocket", k=2, **kwargs)]

    assert first == second


# --------------------------------------------------------------------------- #
# retrieve: edge cases
# --------------------------------------------------------------------------- #


def test_a_missing_collection_is_empty_not_an_error():
    """A fresh install has no collection yet; that is a normal state."""
    client = QdrantClient(":memory:")

    assert retrieve("rocket", client=client, collection="nope", embedder=FakeEmbedder()) == []


def test_an_empty_collection_returns_nothing():
    assert retrieve("rocket", client=_client_with([]), collection=COLLECTION,
                    embedder=FakeEmbedder()) == []


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_a_blank_query_never_touches_qdrant(query):
    embedder = FakeEmbedder()

    assert retrieve(query, client=None, embedder=embedder) == []
    assert embedder.queries == []


def test_k_of_zero_returns_nothing():
    assert retrieve("rocket", k=0, client=None, embedder=FakeEmbedder()) == []


# --------------------------------------------------------------------------- #
# retrieve: embedding-model guard
# --------------------------------------------------------------------------- #


def test_a_collection_written_by_another_model_is_refused():
    """Vectors from two models live in unrelated spaces: the search would still
    return results, ranked by a similarity that means nothing."""
    client = _client_with(
        [_point(1, [1.0, 0.0], title="story", published=NOW, model="some-other-model")]
    )

    with pytest.raises(EmbeddingModelMismatchError, match="some-other-model"):
        retrieve("rocket", client=client, collection=COLLECTION, embedder=FakeEmbedder(), now=NOW)


def test_a_point_without_a_recorded_model_is_still_searchable():
    """Points written before the field existed must not lock a user out of
    their own index."""
    client = _client_with([_point(1, [1.0, 0.0], title="legacy", published=NOW, model=None)])

    hits = retrieve("rocket", client=client, collection=COLLECTION,
                    embedder=FakeEmbedder(), now=NOW)

    assert [h.title for h in hits] == ["legacy"]
