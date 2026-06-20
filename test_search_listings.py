"""
test_search_listings.py

Basic tests for Tool 1 (search_listings) and its helpers in tools.py.

Run directly:        python test_search_listings.py
Or with pytest:      pytest test_search_listings.py
"""

from unittest.mock import patch

import tools
from tools import (
    _keywords,
    _listing_text,
    _size_matches,
    search_listings,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────
# Handcrafted listings so tests assert on KNOWN data, not the live dataset
# (which may change). Only the fields search_listings touches are included.

def _make(id, title, description, style_tags, size, price):
    return {
        "id": id,
        "title": title,
        "description": description,
        "style_tags": style_tags,
        "size": size,
        "price": price,
    }


FAKE_LISTINGS = [
    # Strong match for "vintage graphic tee": hits title + tags.
    _make("a", "Graphic Tee", "a cool shirt",
          ["vintage", "graphic tee"], "S/M", 18.0),
    # Partial match: only "vintage" overlaps.
    _make("b", "Crewneck Sweatshirt", "cozy navy pullover",
          ["vintage", "basics"], "L", 20.0),
    # Zero match: no query keyword appears anywhere → must be dropped.
    _make("c", "Mesh Top", "sheer black layer",
          ["grunge", "goth"], "S/M", 15.0),
    # Over budget: matches keywords but should be filtered out by max_price.
    _make("d", "Vintage Graphic Tee Deluxe", "premium print",
          ["vintage", "graphic tee"], "M", 40.0),
]


# ── Helper tests ─────────────────────────────────────────────────────────────

def test_keywords_strips_stopwords_punct_and_numbers():
    # "i'm", "looking", "for", "a", "under" are stopwords; "30" is a bare digit.
    result = _keywords("I'm looking for a vintage graphic tee under $30")
    assert result == {"vintage", "graphic", "tee"}


def test_keywords_empty_query_is_empty_set():
    assert _keywords("") == set()
    assert _keywords("   ") == set()


def test_listing_text_flattens_tags_and_lowercases():
    words = _listing_text(FAKE_LISTINGS[0])
    # From title "Graphic Tee" and tag list ["vintage", "graphic tee"].
    assert {"graphic", "tee", "vintage"} <= words


def test_size_matches_token_not_substring():
    assert _size_matches("M", "S/M") is True        # docstring's key example
    assert _size_matches("m", "S/M") is True         # case-insensitive
    assert _size_matches("L", "S/M") is False
    # "m" must NOT match the 'm' buried in other words.
    assert _size_matches("M", "One Size (adjustable)") is False
    assert _size_matches("M", "W30 L30") is False


# ── search_listings tests (helpers patched? no — patch the data loader) ───────
# Patch load_listings so the function runs against FAKE_LISTINGS, not the file.

def _run(*args, **kwargs):
    with patch.object(tools, "load_listings", return_value=FAKE_LISTINGS):
        return search_listings(*args, **kwargs)


def test_ranks_by_score_and_drops_zero_match():
    results = _run("vintage graphic tee")
    ids = [r["id"] for r in results]
    # "a" (3 hits) before "b" (1 hit); "c" dropped (0); "d" present (no filter).
    assert ids[0] == "a"
    assert "c" not in ids
    assert results[0]["id"] == "a"


def test_max_price_filters_before_scoring():
    results = _run("vintage graphic tee", max_price=25.0)
    ids = [r["id"] for r in results]
    assert "d" not in ids          # $40 over the $25 ceiling
    assert "a" in ids              # $18 stays


def test_size_filter_uses_token_match():
    results = _run("vintage graphic tee", size="M")
    ids = [r["id"] for r in results]
    # "a" has size "S/M" → token "m" matches; "d" has "M" → matches but $40...
    # (no price filter here) so both qualify on size.
    assert "a" in ids
    assert "d" in ids


def test_no_keyword_overlap_returns_empty_list():
    results = _run("kayak helicopter")     # words nothing in fixtures has
    assert results == []                    # honest empty list, no exception


def test_real_dataset_loads_and_is_searchable():
    # Sanity check against the actual data file (not mocked): a broad query
    # should return a non-empty list of dicts without raising.
    results = search_listings("vintage")
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(r, dict) and "id" in r for r in results)


# ── Plain-python runner (so `python test_search_listings.py` works) ──────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001 — surface any unexpected error
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
        else:
            print(f"ok    {t.__name__}")
            passed += 1
    print(f"\n{passed}/{len(tests)} passed")
