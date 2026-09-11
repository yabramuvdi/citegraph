"""Stable identifier generation for works.

We don't want IDs to change between runs, so they are derived from the
content (first author + year + a title prefix) rather than from row index.
The single ``w-`` prefix is role-free on purpose: a work's role (core
source vs cited stub) can change over its life; its identity must not.
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
    elif not isinstance(authors, str):
        # NaN or other non-string scalars from pandas
        return "unknown"
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


def make_work_id(
    authors: str | list[str],
    year: int | str | None,
    title: str,
) -> str:
    """Build a stable, slug-style work id from bibliographic fields.

    Examples
    --------
    >>> make_work_id("Ostrom, E.", 1990, "Governing the Commons")
    'w-ostrom-1990-governing-the-commons'
    """
    author = _first_author_token(authors)
    year_token = str(year) if year not in (None, "") else "nd"
    title_str = title if isinstance(title, str) else ""
    title_slug = slugify(title_str or "untitled", max_length=40, word_boundary=True) or "untitled"
    return f"w-{author}-{year_token}-{title_slug}"


def assign_source_ids(records: list[dict], previous_registry: dict | None = None) -> tuple[list[dict], dict]:
    """Allocate source observation IDs without dropping collisions or reusing history."""
    from citegraph.io import metadata_fingerprint

    previous = previous_registry or {}
    entries = [dict(e) for e in previous.get("entries", [])]
    reserved = set(previous.get("reserved_ids", [])) | {e["id"] for e in entries}
    lookup = {(e["source_file"], e["metadata_fingerprint"]): e["id"] for e in entries}
    if len({e["id"] for e in entries}) != len(entries):
        raise ValueError("source_ids.json assigns one ID to multiple observations; restore the registry")
    result = [dict(r) for r in records]
    collision_ids = set(previous.get("collision_ids", []))
    # Base IDs are not hashes; sort complete metadata first so allocation is reproducible.
    order = sorted(range(len(result)), key=lambda i: (
        str(result[i].get("Title", "")), str(result[i].get("Authors_List", "")),
        str(result[i].get("Journal", "")), str(result[i].get("Year", "")),
        str(result[i]["source_file"])))
    for i in order:
        rec = result[i]
        signature = metadata_fingerprint(rec)
        key = (str(rec["source_file"]), signature)
        if key in lookup:
            rec["id"] = lookup[key]
            continue
        base = make_work_id(rec.get("Authors_List") or rec.get("Authors", ""), rec.get("Year"), rec.get("Title", ""))
        candidate = base
        suffix = 2
        while candidate in reserved:
            collision_ids.add(base)
            candidate = f"{base}-{suffix}"
            suffix += 1
        if candidate != base:
            collision_ids.add(candidate)
        reserved.add(candidate)
        rec["id"] = candidate
        lookup[key] = candidate
        entries.append(dict(source_file=key[0], metadata_fingerprint=signature, id=candidate))
    return result, dict(schema_version=1, entries=entries, reserved_ids=sorted(reserved),
                        collision_ids=sorted(collision_ids))
