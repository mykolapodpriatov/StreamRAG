"""Tests for the pure ``clean_html_text`` helper in worker.py.

The helper strips HTML tags, decodes entities, and collapses whitespace so RSS
``title``/``summary`` fields do not leak markup into downstream embeddings. All
cases are pure string transforms with no network or feedparser involvement.
"""

import worker


def test_strips_tags_and_keeps_text():
    assert worker.clean_html_text("<p>Hello <b>world</b></p>") == "Hello world"


def test_drops_tag_attributes_but_keeps_link_text():
    assert (
        worker.clean_html_text('<a href="https://example.com">click here</a>')
        == "click here"
    )


def test_decodes_html_entities():
    assert worker.clean_html_text("Fish &amp; Chips &lt;3") == "Fish & Chips <3"


def test_numeric_and_named_charrefs_decode_once():
    # &#38; is the numeric form of & ; the named &amp; is also decoded. Each
    # reference is unescaped exactly once (no double-decoding).
    assert worker.clean_html_text("A &#38; B &amp; C") == "A & B & C"


def test_double_escaped_entity_decodes_only_one_level():
    # &amp;lt; must yield the literal "&lt;", not collapse to "<".
    assert worker.clean_html_text("&amp;lt;tag&amp;gt;") == "&lt;tag&gt;"


def test_collapses_internal_and_surrounding_whitespace():
    assert (
        worker.clean_html_text("  <div>  spaced\n\tout   text  </div>  ")
        == "spaced out text"
    )


def test_empty_input_returns_empty_string():
    assert worker.clean_html_text("") == ""


def test_whitespace_and_tag_only_input_returns_empty_string():
    # Markup with no text content collapses to the empty string.
    assert worker.clean_html_text("  <br/> <hr>  ") == ""


def test_plain_text_is_returned_unchanged():
    # Tagless, entity-free input with single spaces survives verbatim, which is
    # what keeps the extract_entries verbatim-mapping tests green.
    assert (
        worker.clean_html_text("The full summary of the first post.")
        == "The full summary of the first post."
    )
