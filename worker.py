import ipaddress
import logging
import os
import socket
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

        entries = []
        for entry in getattr(feed, "entries", []):
            entries.append({
                "title": getattr(entry, "title", "No Title"),
                "link": getattr(entry, "link", ""),
                "published": entry.get("published", ""),
                "summary": entry.get("summary", "")
            })
            
        # TODO: Clean text, generate embeddings, and index into Qdrant
        return len(entries)
    except Exception:
        # Re-raise so Celery records the task as FAILED (and can retry) instead
        # of masking the error as a successful empty result.
        logger.exception("Error processing feed %s", safe_url)
        raise
