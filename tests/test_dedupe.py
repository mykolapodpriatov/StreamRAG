"""Tests for the pure ``dedupe_entries`` helper in worker.py.

Re-polled feeds re-deliver the same items; ``dedupe_entries`` collapses them by
identity (non-empty ``link``, falling back to ``title``) while preserving the
first-seen order. All cases are plain in-memory dict lists with no I/O.
"""

import worker


def test_duplicate_links_keep_only_first_occurrence():
    entries = [
        {"title": "First", "link": "https://example.com/a"},
        {"title": "Second", "link": "https://example.com/b"},
        {"title": "First again (re-delivered)", "link": "https://example.com/a"},
    ]
    deduped = worker.dedupe_entries(entries)
    assert [e["link"] for e in deduped] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    # The retained entry is the original first occurrence, not the later copy.
    assert deduped[0]["title"] == "First"


def test_distinct_links_are_all_retained():
    entries = [
        {"title": "A", "link": "https://example.com/1"},
        {"title": "B", "link": "https://example.com/2"},
        {"title": "C", "link": "https://example.com/3"},
    ]
    assert worker.dedupe_entries(entries) == entries


def test_empty_link_falls_back_to_title_for_identity():
    entries = [
        {"title": "Shared Title", "link": ""},
        {"title": "Shared Title", "link": ""},
        {"title": "Different Title", "link": ""},
    ]
    deduped = worker.dedupe_entries(entries)
    assert [e["title"] for e in deduped] == ["Shared Title", "Different Title"]


def test_link_takes_precedence_over_title_when_present():
    # Same title but different links => distinct entries, both kept.
    entries = [
        {"title": "Same Title", "link": "https://example.com/x"},
        {"title": "Same Title", "link": "https://example.com/y"},
    ]
    assert worker.dedupe_entries(entries) == entries


def test_order_is_preserved():
    entries = [
        {"title": "T3", "link": "l3"},
        {"title": "T1", "link": "l1"},
        {"title": "T2", "link": "l2"},
        {"title": "T1 dup", "link": "l1"},
        {"title": "T3 dup", "link": "l3"},
    ]
    deduped = worker.dedupe_entries(entries)
    assert [e["link"] for e in deduped] == ["l3", "l1", "l2"]


def test_entries_without_any_identity_are_always_kept():
    # No link and no title => no usable identity, so nothing can be deduped.
    entries = [
        {"title": "", "link": ""},
        {"title": "", "link": ""},
    ]
    assert worker.dedupe_entries(entries) == entries


def test_custom_key_is_honored():
    entries = [
        {"title": "One", "guid": "g1"},
        {"title": "Two", "guid": "g1"},
        {"title": "Three", "guid": "g2"},
    ]
    deduped = worker.dedupe_entries(entries, key="guid")
    assert [e["title"] for e in deduped] == ["One", "Three"]


def test_empty_input_returns_empty_list():
    assert worker.dedupe_entries([]) == []
