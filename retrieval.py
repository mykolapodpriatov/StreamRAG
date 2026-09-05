"""Query-side retrieval over the collection the worker writes.

Deliberately importable without Celery or Streamlit, the same way
:mod:`scoring` is, so the ranking logic can be unit-tested against a fake
embedder and an in-memory Qdrant client with no services running.

Scope is retrieval only: this returns ranked passages, not a generated answer.
Generation needs an LLM key and turns every test into a mock or a network call,
and the interesting decision (how similarity and recency trade off) lives here.

Two choices worth knowing about:

* **Recency is recomputed at query time.** The worker stores a
  ``recency_weight`` on each point, but it was computed when the entry was
  indexed and decays out of date from that moment. Ranking reads ``published``
  and re-derives the weight against *now*.
* **The candidate pool is wider than the answer.** Blending recency in can
  promote an entry that was not in the top-k by similarity, so Qdrant is asked
  for ``k * CANDIDATE_MULTIPLIER`` hits and the blend re-ranks them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from config import (
    CANDIDATE_MULTIPLIER,
    EMBEDDING_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    RECENCY_WEIGHT,
)
from scoring import recency_weight

__all__ = [
    "EmbeddingModelMismatchError",
    "RetrievedPassage",
    "blend_score",
    "retrieve",
]


class EmbeddingModelMismatchError(RuntimeError):
    """The collection was written by a different embedding model than we query with.

    This is fatal rather than a warning. Vectors from two different models live
    in unrelated spaces, so the search still returns results, ranked by a
    similarity that means nothing. A loud failure is the only way that gets
    noticed.
    """


@dataclass(frozen=True)
class RetrievedPassage:
    """One ranked hit, with everything a reader needs to judge it.

    ``link`` and ``published`` are part of the result rather than an extra
    lookup: a news passage without a source and a date is not usable.
    """

    title: str
    link: str
    published: str
    summary: str
    similarity: float
    recency: float
    score: float


def blend_score(similarity: float, recency: float, weight: float) -> float:
    """Combine a similarity and a recency weight into one ranking score.

    ``weight`` is recency's share of the result, in ``[0, 1]``: 0 ranks purely
    by similarity, 1 purely by recency. Both inputs are already normalised to
    ``[0, 1]`` (cosine similarity from Qdrant, and
    :func:`scoring.recency_weight`), so the blend needs no rescaling.

    An entry with no usable date gets ``scoring.UNKNOWN_DATE_WEIGHT`` (0.0),
    which costs it the recency share of its score but does not exclude it. A
    strong undated match can still outrank a weak recent one, which is the
    behaviour you want: an unparseable date is a gap in the feed's metadata,
    not evidence that the entry is old.

    Raises:
        ValueError: if ``weight`` is outside ``[0, 1]``.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"recency weight must be in [0, 1], got {weight!r}")
    return (1.0 - weight) * similarity + weight * recency


def _default_client():
    """Build a Qdrant client from config, imported lazily.

    Kept out of module scope so importing this module does not require the
    qdrant client to be installed or a server to be reachable, matching how
    worker.py defers its own clients.
    """
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_URL, check_compatibility=False)


def _default_embedder():
    """Build the query embedder from config, imported lazily.

    Instantiating it eagerly would require OPENAI_API_KEY just to import this
    module, which would make the offline test suite depend on a key it has no
    use for.
    """
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def _check_model(payload: dict) -> None:
    """Fail if a hit was written by a different embedding model.

    A point with no recorded model predates the field. It is left alone rather
    than rejected: refusing to search an older collection would be a worse
    failure than the one this guards against.
    """
    written_with = payload.get("embedding_model")
    if written_with and written_with != EMBEDDING_MODEL:
        raise EmbeddingModelMismatchError(
            f"collection was written with embedding model {written_with!r} but this process "
            f"queries with {EMBEDDING_MODEL!r}; the vectors are not comparable. Re-index the "
            "collection, or set EMBEDDING_MODEL to match."
        )


def retrieve(
    query: str,
    k: int = 5,
    *,
    client=None,
    collection: str = QDRANT_COLLECTION,
    embedder=None,
    weight: float = RECENCY_WEIGHT,
    now: datetime | None = None,
    candidate_multiplier: int = CANDIDATE_MULTIPLIER,
) -> list[RetrievedPassage]:
    """Return the ``k`` best passages for ``query``, newest-and-closest first.

    Args:
        query: Free-text query. Blank returns no hits without touching Qdrant.
        k: How many passages to return. ``<= 0`` returns nothing.
        client: Qdrant client override, for tests.
        collection: Collection to search.
        embedder: Anything with ``embed_query(str) -> list[float]``. Overridden
            in tests so nothing calls a real embedding API.
        weight: Recency's share of the score; see :func:`blend_score`.
        now: Clock override forwarded to :func:`scoring.recency_weight`, so a
            test can pin what "recent" means.
        candidate_multiplier: How many candidates to fetch per requested result.

    Returns:
        Ranked passages, best first. An empty list when the query is blank, the
        collection does not exist, or nothing has been indexed yet. A missing
        collection is a normal state on a fresh install, not an error.

    Raises:
        EmbeddingModelMismatchError: if the collection records a different
            embedding model than this process is configured with.
    """
    if k <= 0 or not query.strip():
        return []

    client = client or _default_client()
    if not client.collection_exists(collection):
        return []

    embedder = embedder or _default_embedder()
    vector = list(embedder.embed_query(query))

    limit = max(k, k * max(candidate_multiplier, 1))
    hits = client.query_points(
        collection_name=collection,
        query=vector,
        limit=limit,
        with_payload=True,
    ).points

    passages: list[RetrievedPassage] = []
    for hit in hits:
        payload = dict(hit.payload or {})
        _check_model(payload)
        published = str(payload.get("published", ""))
        recency = recency_weight(published, now=now)
        similarity = float(hit.score)
        passages.append(
            RetrievedPassage(
                title=str(payload.get("title", "")),
                link=str(payload.get("link", "")),
                published=published,
                summary=str(payload.get("summary", "")),
                similarity=similarity,
                recency=recency,
                score=blend_score(similarity, recency, weight),
            )
        )

    # Ties broken by link so two runs over the same data agree.
    passages.sort(key=lambda p: (-p.score, p.link))
    return passages[:k]
