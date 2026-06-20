"""
tools.py

The three required FitFindr tools. Each tool is a standalone function that
can be called and tested independently before being wired into the agent loop.

Complete and test each tool before moving to agent.py.

Tools:
    search_listings(description, size, max_price)  → list[dict]
    suggest_outfit(new_item, wardrobe)              → str
    create_fit_card(outfit, new_item)               → str
"""

import os
import re

from dotenv import load_dotenv
from groq import Groq

from utils.data_loader import load_listings

load_dotenv()


# ── search_listings config ──────────────────────────────────────────────────

# Common function words stripped from the query so they don't count as keywords.
STOPWORDS = {
    "i", "im", "a", "an", "the", "for", "with", "under",
    "and", "or", "to", "in", "of", "my", "looking",
}

# Listing fields a keyword can match against (high-signal, always string-ish).
SCORE_FIELDS = ("title", "description", "style_tags")


# ── Groq client ───────────────────────────────────────────────────────────────

def _get_groq_client():
    """Initialize and return a Groq client using GROQ_API_KEY from .env."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to a .env file in the project root."
        )
    return Groq(api_key=api_key)


# ── Tool 1: search_listings ───────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Lowercase `text` and return the set of word tokens (letters/digits)."""
    # CRITICAL: same tokenizer is used for BOTH the query and listing text, so
    # the two sides normalize identically (punctuation/case can't cause misses).
    # [a-z0-9]+ grabs runs of letters/digits → punctuation falls away.
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _keywords(description: str) -> set[str]:
    """Turn a query into a keyword set: tokenize, drop stopwords and bare digits."""
    return {
        word
        for word in _tokenize(description)
        # Drop function words ("for", "a") so they can't inflate scores (1c),
        # and bare numbers ("30" from "under $30") — price is handled separately.
        if word not in STOPWORDS and not word.isdigit()
        # CRITICAL: drop single letters. Contractions split on the apostrophe
        # ("I'm" → "i","m"), leaving debris like "m"/"s"/"t" that aren't real
        # keywords. Size is its own parameter, so 1-char tokens never help here.
        and len(word) > 1
    }


def _listing_text(listing: dict) -> set[str]:
    """Collect the searchable word tokens from a listing's SCORE_FIELDS."""
    parts: list[str] = []
    for field in SCORE_FIELDS:
        value = listing.get(field)
        # CRITICAL: style_tags is a list — flatten it; other fields are strings.
        if isinstance(value, list):
            parts.extend(str(v) for v in value)
        elif value is not None:
            parts.append(str(value))
    return _tokenize(" ".join(parts))


def _size_matches(query_size: str, listing_size: str) -> bool:
    """True if `query_size` is one of the listing size's tokens (case-insensitive).

    Splits on non-alphanumeric chars so "M" matches "S/M" but not the "m" in
    "One Size (adjustable)".
    """
    # CRITICAL: token match, NOT substring — splitting "S/M" → ["s","m"] lets
    # "M" match while avoiding false positives (e.g. "m" inside other words).
    tokens = re.split(r"[^a-z0-9]+", listing_size.lower())
    return query_size.lower() in tokens


def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    """
    Search the mock listings dataset for items matching the description,
    optional size, and optional price ceiling.

    Args:
        description: Keywords describing what the user is looking for
                     (e.g., "vintage graphic tee").
        size:        Size string to filter by, or None to skip size filtering.
                     Matching is case-insensitive (e.g., "M" matches "S/M").
        max_price:   Maximum price (inclusive), or None to skip price filtering.

    Returns:
        A list of matching listing dicts, sorted by relevance (best match first).
        Returns an empty list if nothing matches — does NOT raise an exception.

    Each listing dict has the following fields:
        id, title, description, category, style_tags (list), size,
        condition, price (float), colors (list), brand, platform

    TODO:
        1. Load all listings with load_listings().
        2. Filter by max_price and size (if provided).
        3. Score each remaining listing by keyword overlap with `description`.
        4. Drop any listings with a score of 0 (no relevant matches).
        5. Sort by score, highest first, and return the listing dicts.

    Before writing code, fill in the Tool 1 section of planning.md.
    """
    # 1. Load every listing from the mock dataset.
    listings = load_listings()

    # 2. Reduce the query to meaningful keywords once, up front (1c).
    keywords = _keywords(description)

    scored: list[tuple[int, dict]] = []
    for listing in listings:
        # ---- FILTER FIRST (cheap guards before any scoring) ----
        # Price ceiling is inclusive; skip the check entirely when not given.
        if max_price is not None and listing["price"] > max_price:
            continue
        # Size filter uses token matching ("M" ⊆ "S/M"); skip when not given.
        if size is not None and not _size_matches(size, listing["size"]):
            continue

        # ---- SCORE by keyword overlap (set intersection = shared words) ----
        score = len(keywords & _listing_text(listing))

        # 4. Drop anything with no keyword overlap — a kept item shares ≥1 word.
        if score == 0:
            continue

        scored.append((score, listing))

    # 5. Highest score first. Python's sort is STABLE, so ties keep the
    #    original dataset order (no secondary sort key needed).
    scored.sort(key=lambda pair: pair[0], reverse=True)

    # Hand back just the listing dicts, untouched (score stays out of the data).
    return [listing for _, listing in scored]


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────

def suggest_outfit(new_item: dict, wardrobe: dict) -> str:
    """
    Given a thrifted item and the user's wardrobe, suggest 1–2 complete outfits.

    Args:
        new_item: A listing dict (the item the user is considering buying).
        wardrobe: A wardrobe dict with an 'items' key containing a list of
                  wardrobe item dicts. May be empty — handle this gracefully.

    Returns:
        A non-empty string with outfit suggestions.
        If the wardrobe is empty, offer general styling advice for the item
        rather than raising an exception or returning an empty string.

    TODO:
        1. Check whether wardrobe['items'] is empty.
        2. If empty: call the LLM with a prompt for general styling ideas
           (what kinds of items pair well, what vibe it suits, etc.).
        3. If not empty: format the wardrobe items into a prompt and ask
           the LLM to suggest specific outfit combinations using the new item
           and named pieces from the wardrobe.
        4. Return the LLM's response as a string.

    Before writing code, fill in the Tool 2 section of planning.md.
    """
    # Replace this with your implementation
    return ""


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def create_fit_card(outfit: str, new_item: dict) -> str:
    """
    Generate a short, shareable outfit caption for the thrifted find.

    Args:
        outfit:   The outfit suggestion string from suggest_outfit().
        new_item: The listing dict for the thrifted item.

    Returns:
        A 2–4 sentence string usable as an Instagram/TikTok caption.
        If outfit is empty or missing, return a descriptive error message
        string — do NOT raise an exception.

    The caption should:
    - Feel casual and authentic (like a real OOTD post, not a product description)
    - Mention the item name, price, and platform naturally (once each)
    - Capture the outfit vibe in specific terms
    - Sound different each time for different inputs (use higher LLM temperature)

    TODO:
        1. Guard against an empty or whitespace-only outfit string.
        2. Build a prompt that gives the LLM the item details and the outfit,
           and asks for a caption matching the style guidelines above.
        3. Call the LLM and return the response.

    Before writing code, fill in the Tool 3 section of planning.md.
    """
    # Replace this with your implementation
    return ""
