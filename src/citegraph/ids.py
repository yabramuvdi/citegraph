"""Stable identifier generation for papers and references.

We don't want IDs to change between runs, so they are derived from the
content (first author + year + a title prefix) rather than from row index.
"""

from __future__ import annotations

import re

from slugify import slugify

_NON_ALPHA = re.compile(r"[^a-zA-Z]+")


def _first_author_token(authors: str | list[str]) -> str:
    """Return a single lower-case surname token for the first author.

    Handles both ``"Ostrom, E."`` (last-name-first) and ``"Elinor Ostrom"``
    (first-name-first) styles. For multi-author lists, only the first
    author is considered.
    """
    if isinstance(authors, list):
        first = authors[0] if authors else ""
    else:
        first = authors.split(";")[0] if authors else ""
        if "," not in first and " and " in first:
            first = first.split(" and ")[0]

    first = first.strip()
    if not first:
        return "unknown"

    if "," in first:
        surname = first.split(",")[0]
    else:
        tokens = [t for t in _NON_ALPHA.split(first) if t]
        surname = tokens[-1] if tokens else ""

    surname = "".join(ch for ch in surname if ch.isalpha())
    return surname.lower() or "unknown"


def make_paper_id(
    authors: str | list[str],
    year: int | str | None,
    title: str,
    *,
    prefix: str = "p",
) -> str:
    """Build a stable, slug-style id from bibliographic fields.

    Examples
    --------
    >>> make_paper_id("Ostrom, E.", 1990, "Governing the Commons")
    'p-ostrom-1990-governing-the-commons'
    """
    author = _first_author_token(authors)
    year_token = str(year) if year not in (None, "") else "nd"
    title_slug = slugify(title or "untitled", max_length=40, word_boundary=True) or "untitled"
    return f"{prefix}-{author}-{year_token}-{title_slug}"


def make_reference_id(
    authors: str | list[str],
    year: int | str | None,
    title: str,
) -> str:
    """Like :func:`make_paper_id` but with an ``r-`` prefix for references."""
    return make_paper_id(authors, year, title, prefix="r")
