import html
import ipaddress
import logging
import os
import socket
from html.parser import HTMLParser
from urllib.parse import urlparse, urlunparse

from celery import Celery
import feedparser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery("streamrag", broker=REDIS_URL, backend=REDIS_URL)

# Bound the network fetch performed by feedparser so a slow/hanging feed server
# cannot block the Celery worker indefinitely.
FEED_FETCH_TIMEOUT = float(os.getenv("FEED_FETCH_TIMEOUT", "10"))


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
    """Fetch and process an RSS feed and return the number of entries found."""
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

        entries = extract_entries(feed)

        # TODO: Clean text, generate embeddings, and index into Qdrant
        return len(entries)
    except Exception:
        # Re-raise so Celery records the task as FAILED (and can retry) instead
        # of masking the error as a successful empty result.
        logger.exception("Error processing feed %s", safe_url)
        raise
