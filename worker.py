import html
import ipaddress
import logging
import os
import socket
import uuid
from html.parser import HTMLParser
from urllib.parse import urlparse, urlunparse

from celery import Celery
import feedparser
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery("streamrag", broker=REDIS_URL, backend=REDIS_URL)

# Bound the network fetch performed by feedparser so a slow/hanging feed server
# cannot block the Celery worker indefinitely.
FEED_FETCH_TIMEOUT = float(os.getenv("FEED_FETCH_TIMEOUT", "10"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "streamrag_entries")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# check_compatibility=False: by default the constructor makes a lightweight
# call to the server to compare client/server versions, which would make
# `import worker` (and therefore test collection) depend on a reachable
# Qdrant instance. Actual requests still connect lazily on first use.
qdrant = QdrantClient(url=QDRANT_URL, check_compatibility=False)

# The OpenAI embeddings client is *not* built at import time: instantiating
# it eagerly requires OPENAI_API_KEY to already be set, which would make
# `import worker` (and therefore test collection) fail in any environment
# without that key configured. It is created lazily on first use instead.
_embeddings: OpenAIEmbeddings | None = None


def _get_embeddings() -> OpenAIEmbeddings:
    """Lazily construct and cache the OpenAI embeddings client."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings


def _redact_url(url: str) -> str:
    """Strip any userinfo (user:pass@) from a URL before it is logged."""
    try:
        parts = urlparse(url)
    except ValueError:
        return "<unparseable-url>"
    netloc = parts.hostname or ""
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunparse(parts._replace(netloc=netloc))


def _assert_public_host(feed_url: str) -> None:
    """Reject URLs whose host resolves to a private/loopback/link-local address.

    Scheme validation alone does not stop SSRF: an attacker can still point an
    http(s) URL at 127.0.0.1, 10.0.0.0/8, or the cloud metadata endpoint
    (169.254.169.254). We resolve every address the host maps to and refuse if
    any is non-global. (Note: feedparser follows redirects internally, so a
    fully hardened fetcher would also revalidate each redirect hop; that is a
    larger refactor and is tracked separately.)
    """
    host = urlparse(feed_url).hostname
    if not host:
        raise ValueError("Feed URL has no host")
    try:
        addrinfos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve feed host: {host}") from exc
    for family, _, _, _, sockaddr in addrinfos:
        ip = ipaddress.ip_address(sockaddr[0])
        if not ip.is_global:
            raise ValueError(
                f"Refusing to fetch feed pointing at non-public address {ip} (host {host})"
            )


class _TagStripper(HTMLParser):
    """Collect an element's text content while discarding tags and attributes.

    ``convert_charrefs`` is disabled so the parser does not decode character or
    entity references piecemeal; instead they are reassembled verbatim and left
    for a single ``html.unescape`` pass in :func:`clean_html_text`. That keeps
    every reference decoded exactly once (so e.g. ``&amp;lt;`` yields ``&lt;``,
    not ``<``).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def handle_entityref(self, name: str) -> None:
        self._parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._parts.append(f"&#{name};")

    def get_text(self) -> str:
        return "".join(self._parts)


def clean_html_text(value: str) -> str:
    """Strip HTML tags, decode entities, and collapse whitespace in feed text.

    Pure and side-effect free. RSS ``title``/``summary`` fields routinely carry
    markup (``<p>``, ``<a href=...>``) and HTML entities (``&amp;``) that would
    otherwise pollute downstream embeddings. Tags are removed, references are
    unescaped once, and runs of whitespace collapse to single spaces with the
    ends trimmed. Plain, tagless input is therefore returned unchanged apart
    from whitespace normalization, preserving the verbatim-mapping guarantees.
    """
    if not value:
        return ""
    stripper = _TagStripper()
    stripper.feed(value)
    stripper.close()
    text = html.unescape(stripper.get_text())
    return " ".join(text.split())


def extract_entries(feed) -> list[dict]:
    """Map a parsed feedparser result into a list of normalized entry dicts.

    Pure and side-effect free: it performs no network I/O, so it can be tested
    directly against ``feedparser.parse(<raw string>)`` output. Missing fields
    fall back to safe defaults ("No Title" for the title, "" for link,
    published and summary) so downstream indexing never encounters a missing
    key. A malformed (bozo) feed still yields whatever entries feedparser
    managed to recover, because this function only reads ``feed.entries``.
    """
    entries: list[dict] = []
    for entry in getattr(feed, "entries", []):
        entries.append({
            "title": clean_html_text(entry.get("title", "No Title")),
            "link": entry.get("link", ""),
            "published": entry.get("published", ""),
            "summary": clean_html_text(entry.get("summary", "")),
        })
    return entries


def dedupe_entries(entries: list[dict], key: str = "link") -> list[dict]:
    """Drop entries already seen under their identity, preserving first-seen order.

    Pure and side-effect free. Streaming re-polls the same feeds, so
    :func:`extract_entries` yields the same items on every run; re-embedding
    them into Qdrant each time is wasteful. Each entry's identity is its
    non-empty ``key`` field (``"link"`` by default), falling back to ``"title"``
    when the key is missing or empty. The first occurrence of an identity is
    kept and any later entry sharing it is discarded. Entries with no usable
    identity (empty ``key`` and empty ``title``) cannot be compared and are
    always kept, so genuinely distinct-but-anonymous items are never merged.
    """
    seen: set[str] = set()
    deduped: list[dict] = []
    for entry in entries:
        identity = entry.get(key) or entry.get("title") or ""
        if identity:
            if identity in seen:
                continue
            seen.add(identity)
        deduped.append(entry)
    return deduped


def _point_id_for_entry(entry: dict) -> str:
    """Derive a stable Qdrant point ID from an entry's identity.

    Hashed deterministically (uuid5 over a fixed namespace) from the entry's
    ``link`` so that re-polling the same feed re-embeds and *upserts* the same
    point instead of accumulating duplicates. Falls back to ``title`` for the
    rare entry with no link, mirroring :func:`dedupe_entries`'s identity rule
    so two different no-link entries don't collide on the same empty-string
    hash.
    """
    identity = entry.get("link") or entry.get("title") or ""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def _ensure_collection(vector_size: int) -> None:
    """Create the target Qdrant collection on first use, idempotently.

    Safe to call before every batch of upserts: it only issues a
    ``create_collection`` call when the collection does not already exist.
    Cosine distance matches the metric OpenAI's embedding models are tuned
    for.
    """
    if not qdrant.collection_exists(QDRANT_COLLECTION):
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def _embed_and_index(entries: list[dict]) -> int:
    """Embed each entry and upsert it into Qdrant, returning the point count.

    An empty ``entries`` list is a no-op (no embedding call, no collection
    lookup). Any failure from the embedding call or the Qdrant upsert
    propagates to the caller unchanged; nothing here swallows exceptions.
    """
    if not entries:
        return 0

    texts = [
        f"{entry.get('title', '')}\n\n{entry.get('summary', '')}".strip() for entry in entries
    ]
    vectors = _get_embeddings().embed_documents(texts)

    _ensure_collection(len(vectors[0]))

    points = [
        PointStruct(
            id=_point_id_for_entry(entry),
            vector=vector,
            payload={
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "summary": entry.get("summary", ""),
            },
        )
        for entry, vector in zip(entries, vectors)
    ]
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)


def _parse_feed_with_timeout(feed_url: str):
    """Run feedparser.parse with a bounded socket timeout.

    feedparser performs the HTTP fetch itself via urllib, which has no default
    timeout; without this a slow or hanging server would block the worker.
    """
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(FEED_FETCH_TIMEOUT)
    try:
        return feedparser.parse(feed_url)
    finally:
        socket.setdefaulttimeout(previous_timeout)


@app.task
def process_rss_feed(feed_url: str):
    """Fetch, dedupe, embed, and index an RSS feed's entries into Qdrant.

    Returns the number of points actually upserted into the collection.
    """
    safe_url = _redact_url(feed_url)

    # Reject non-HTTP(S) schemes to mitigate SSRF / local-file disclosure
    # (e.g. file://, ftp://) via crafted feed URLs.
    scheme = urlparse(feed_url).scheme.lower()
    if scheme not in ("http", "https"):
        logger.error("Refusing to fetch feed with unsupported scheme %r: %s", scheme, safe_url)
        raise ValueError(f"Unsupported feed URL scheme: {scheme!r}")

    # Block hosts that resolve to private/loopback/link-local addresses (SSRF).
    _assert_public_host(feed_url)

    try:
        feed = _parse_feed_with_timeout(feed_url)

        if getattr(feed, "bozo", False):
            # feedparser sets the bozo flag if it encounters a badly formatted feed
            logger.warning("Poorly formatted feed %s", safe_url)

        entries = dedupe_entries(extract_entries(feed))

        return _embed_and_index(entries)
    except Exception:
        # Re-raise so Celery records the task as FAILED (and can retry) instead
        # of masking the error as a successful empty result.
        logger.exception("Error processing feed %s", safe_url)
        raise
