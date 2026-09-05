"""Streamlit dashboard for StreamRAG ingestion metrics.

Qdrant/Redis helpers live at module scope so they can be unit-tested without
importing Streamlit (CI does not install it) or talking to live services.
"""

from celery import Celery
from qdrant_client import QdrantClient
import redis as redis_lib

import config
from retrieval import EmbeddingModelMismatchError, retrieve

# Read from config so the dashboard, the worker and the retriever cannot drift
# apart on the collection they point at.
QDRANT_URL = config.QDRANT_URL
QDRANT_COLLECTION = config.QDRANT_COLLECTION
REDIS_URL = config.REDIS_URL

#: How many passages the search box shows.
SEARCH_RESULTS = 5

# Shown when Qdrant or Redis cannot be reached. Distinct from ``0``, which
# means "connected, but nothing indexed / no workers yet".
NOT_CONNECTED = "not connected"

# Bound inspect/ping so a down broker cannot hang the dashboard render.
_INSPECT_TIMEOUT = 0.5
_REDIS_SOCKET_TIMEOUT = 0.5


def metric_or_fallback(
    *,
    value: int | None = None,
    missing: bool = False,
    error: bool = False,
) -> int | str:
    """Map a live reading onto a dashboard-safe value.

    Pure: no I/O. Connection failures become :data:`NOT_CONNECTED`; a missing
    collection, a ``None`` inspect reply, or a missing count become ``0``.
    """
    if error:
        return NOT_CONNECTED
    if missing or value is None:
        return 0
    return value


def fetch_document_count(
    client,
    collection: str = QDRANT_COLLECTION,
) -> int | str:
    """Return the collection's point count, or a fallback if Qdrant is unusable.

    ``client`` is injected so tests can exercise missing-collection and
    connection-error paths without a live Qdrant instance.
    """
    try:
        if not client.collection_exists(collection):
            return metric_or_fallback(missing=True)
        info = client.get_collection(collection)
        count = getattr(info, "points_count", None)
        return metric_or_fallback(value=None if count is None else int(count))
    except Exception:
        return metric_or_fallback(error=True)


def fetch_active_streams(redis_client, inspect_active) -> int | str:
    """Return the number of Celery workers, or a fallback if Redis is down.

    Pings Redis first so a dead broker is reported as not connected rather
    than waiting on Celery inspect. ``inspect_active`` should return the
    ``inspect().active()`` mapping (``{worker: [tasks...]}``) or ``None``.
    Both callables are injected so tests never open a real Redis connection.
    """
    try:
        redis_client.ping()
    except Exception:
        return metric_or_fallback(error=True)
    try:
        result = inspect_active()
    except Exception:
        result = None
    if result is None:
        return metric_or_fallback(missing=True)
    return metric_or_fallback(value=len(result))


def _qdrant_client() -> QdrantClient:
    # check_compatibility=False avoids a version handshake on construct.
    return QdrantClient(url=QDRANT_URL, check_compatibility=False)


def _redis_client():
    return redis_lib.from_url(
        REDIS_URL,
        socket_connect_timeout=_REDIS_SOCKET_TIMEOUT,
        socket_timeout=_REDIS_SOCKET_TIMEOUT,
    )


def _celery_inspect_active():
    celery_app = Celery("streamrag", broker=REDIS_URL, backend=REDIS_URL)
    return celery_app.control.inspect(timeout=_INSPECT_TIMEOUT).active()


def load_dashboard_metrics() -> tuple[int | str, int | str]:
    """Fetch both dashboard figures using the process environment."""
    documents = fetch_document_count(_qdrant_client(), QDRANT_COLLECTION)
    streams = fetch_active_streams(_redis_client(), _celery_inspect_active)
    return documents, streams


def main():
    import streamlit as st

    st.set_page_config(page_title="StreamRAG Dashboard", layout="wide")

    st.title("StreamRAG Dashboard")
    st.write("Monitoring streaming data ingestion and RAG system.")

    documents, streams = load_dashboard_metrics()

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Documents", value=documents)
    with col2:
        st.metric(label="Active Streams", value=streams)

    st.subheader("Search the indexed feeds")
    st.caption(
        "These are the passages retrieved from the index, ranked by semantic "
        "similarity blended with recency. They are not a generated answer."
    )
    query = st.text_input("Enter your query:")
    if query:
        with st.spinner("Searching..."):
            try:
                passages = retrieve(query, k=SEARCH_RESULTS, client=_qdrant_client())
            except EmbeddingModelMismatchError as exc:
                st.error(str(exc))
                passages = []
            except Exception as exc:  # Qdrant unreachable, embedder misconfigured
                st.error(f"Search failed: {exc}")
                passages = []

        if not passages:
            st.info("Nothing matched. If the index is empty, run the ingestion worker first.")
        for passage in passages:
            title = passage.title or passage.link or "(untitled)"
            with st.expander(f"{title} · {passage.published or 'no date'}"):
                st.caption(
                    f"score {passage.score:.3f} "
                    f"(similarity {passage.similarity:.3f}, recency {passage.recency:.3f})"
                )
                if passage.link:
                    st.markdown(f"[{passage.link}]({passage.link})")
                if passage.summary:
                    st.write(passage.summary)


if __name__ == "__main__":
    main()
