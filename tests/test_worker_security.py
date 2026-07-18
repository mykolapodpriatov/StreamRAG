"""Offline coverage for worker.py's SSRF guard and URL redaction.

Every test monkeypatches ``socket.getaddrinfo`` so name resolution is fully
simulated: no DNS lookup or network socket is ever opened, which keeps the
suite deterministic and CI-friendly.
"""

import socket

import pytest

import worker


def _addrinfo(*ips: str) -> list[tuple]:
    """Build a ``socket.getaddrinfo``-shaped result from bare IP strings.

    ``_assert_public_host`` only reads ``family`` and ``sockaddr[0]``, so the
    remaining tuple fields are filled with harmless placeholders. IPv6 literals
    (detected by a colon) get the 4-tuple sockaddr the stdlib returns.
    """
    infos: list[tuple] = []
    for ip in ips:
        if ":" in ip:
            family, sockaddr = socket.AF_INET6, (ip, 0, 0, 0)
        else:
            family, sockaddr = socket.AF_INET, (ip, 0)
        infos.append((family, socket.SOCK_STREAM, 0, "", sockaddr))
    return infos


def _patch_resolution(monkeypatch: pytest.MonkeyPatch, *ips: str) -> None:
    """Force host resolution to yield exactly ``ips``."""

    def fake_getaddrinfo(host, port, *args, **kwargs):  # noqa: ANN001, ANN002
        return _addrinfo(*ips)

    monkeypatch.setattr(worker.socket, "getaddrinfo", fake_getaddrinfo)


# --------------------------------------------------------------------------- #
# _assert_public_host
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "private_ip",
    [
        "127.0.0.1",          # IPv4 loopback
        "10.0.0.1",           # RFC1918 10.0.0.0/8 (low end)
        "10.200.30.40",       # RFC1918 10.0.0.0/8 (mid range)
        "169.254.169.254",    # cloud metadata / link-local
        "::1",                # IPv6 loopback
    ],
)
def test_assert_public_host_rejects_non_global(monkeypatch, private_ip):
    _patch_resolution(monkeypatch, private_ip)
    with pytest.raises(ValueError, match="non-public address"):
        worker._assert_public_host("http://internal.example.com/feed")


@pytest.mark.parametrize(
    "public_ip",
    [
        "8.8.8.8",                    # public IPv4
        "1.1.1.1",                    # public IPv4
        "2606:4700:4700::1111",       # global-unicast IPv6
    ],
)
def test_assert_public_host_accepts_global(monkeypatch, public_ip):
    _patch_resolution(monkeypatch, public_ip)
    # A public host must pass silently (the function returns None).
    assert worker._assert_public_host("http://example.com/feed") is None


def test_assert_public_host_rejects_if_any_address_non_global(monkeypatch):
    # DNS-rebinding style: one global and one private address behind one host.
    _patch_resolution(monkeypatch, "8.8.8.8", "10.0.0.1")
    with pytest.raises(ValueError, match="non-public address"):
        worker._assert_public_host("http://rebind.example.com/feed")


def test_assert_public_host_unresolvable_raises_valueerror(monkeypatch):
    def boom(host, port, *args, **kwargs):  # noqa: ANN001, ANN002
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(worker.socket, "getaddrinfo", boom)
    with pytest.raises(ValueError, match="Could not resolve feed host"):
        worker._assert_public_host("http://does-not-exist.invalid/feed")


def test_assert_public_host_without_host_raises_valueerror(monkeypatch):
    # Resolution must never even be attempted when there is no host to resolve.
    def never(*args, **kwargs):
        raise AssertionError("getaddrinfo must not be called when host is empty")

    monkeypatch.setattr(worker.socket, "getaddrinfo", never)
    with pytest.raises(ValueError, match="no host"):
        worker._assert_public_host("http:///path-only")


# --------------------------------------------------------------------------- #
# process_rss_feed scheme gate (runs before any DNS resolution)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad_url",
    ["file:///etc/passwd", "ftp://ftp.example.com/pub/feed.xml"],
)
def test_process_rss_feed_rejects_non_http_scheme_before_resolution(monkeypatch, bad_url):
    def fail_resolution(*args, **kwargs):
        raise AssertionError("getaddrinfo must not run for a rejected scheme")

    monkeypatch.setattr(worker.socket, "getaddrinfo", fail_resolution)
    with pytest.raises(ValueError, match="Unsupported feed URL scheme"):
        # Direct call executes the Celery task body synchronously (no broker).
        worker.process_rss_feed(bad_url)


# --------------------------------------------------------------------------- #
# _redact_url
# --------------------------------------------------------------------------- #
def test_redact_url_strips_credentials():
    redacted = worker._redact_url("https://user:secret@example.com/feed.xml")
    assert redacted == "https://example.com/feed.xml"
    assert "user" not in redacted
    assert "secret" not in redacted


def test_redact_url_preserves_port():
    redacted = worker._redact_url("https://user:pass@example.com:8443/feed")
    assert redacted == "https://example.com:8443/feed"


def test_redact_url_unparseable_returns_sentinel():
    # An unmatched IPv6 bracket makes urlparse itself raise ValueError, which
    # the function catches and reports with its documented sentinel.
    assert worker._redact_url("http://[::1") == "<unparseable-url>"
