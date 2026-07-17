"""Tests for the pure ``extract_entries`` mapping in worker.py.

``feedparser.parse`` accepts a raw string, so these tests parse a local
fixture (and inline malformed markup) with no network access at all.
"""

from pathlib import Path

import feedparser

import worker

FIXTURE = Path(__file__).parent / "fixtures" / "sample_feed.xml"


def _parse_fixture():
    return feedparser.parse(FIXTURE.read_text(encoding="utf-8"))


def test_complete_entry_is_mapped_verbatim():
    entries = worker.extract_entries(_parse_fixture())

    assert entries[0] == {
        "title": "Complete Entry",
        "link": "https://example.com/posts/1",
        "published": "Mon, 06 Sep 2021 12:00:00 GMT",
        "summary": "The full summary of the first post.",
    }


def test_missing_fields_fall_back_to_defaults():
    entries = worker.extract_entries(_parse_fixture())

    # The second fixture item carries only a guid: title, link, published and
    # summary are all absent and must resolve to their documented defaults.
    assert entries[1] == {
        "title": "No Title",
        "link": "",
        "published": "",
        "summary": "",
    }


def test_extract_entries_returns_one_dict_per_item():
    entries = worker.extract_entries(_parse_fixture())

    assert len(entries) == 2
    assert all(
        set(e) == {"title", "link", "published", "summary"} for e in entries
    )


def test_empty_or_entryless_feed_yields_empty_list():
    # A parsed object with no entries must produce an empty list, not an error.
    assert worker.extract_entries(feedparser.parse("<rss></rss>")) == []


def test_malformed_feed_still_yields_recoverable_entries():
    malformed = (
        '<?xml version="1.0"?>\n'
        '<rss version="2.0"><channel><title>Broken</title>\n'
        "<item><title>Recovered Item</title>"
        "<link>https://example.com/x</link></item>\n"
        "<item><title>Unclosed Title</channel></rss>"
    )
    feed = feedparser.parse(malformed)

    # feedparser flags malformed input via a truthy bozo attribute...
    assert feed.bozo
    entries = worker.extract_entries(feed)
    # ...yet the entries it could recover are still mapped normally.
    assert len(entries) >= 1
    assert entries[0]["title"] == "Recovered Item"
    assert entries[0]["link"] == "https://example.com/x"
