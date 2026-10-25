"""JEL classification codes of the core papers.

Most economics papers print their JEL codes on the title page, and docling
carries that line into the markdown, so the codes are read back with a
regex rather than another LLM call: adding a field to ``PaperMetadata``
would invalidate the metadata cache, and re-extracting every paper could
shift titles, hence work ids, hence every id-keyed curated file. Papers
whose PDF carries no codes (most working papers and chapters) get them by hand through
the ``JEL_Codes`` column of ``work_annotations.csv``; a hand-assigned value
overrides extraction, because a reviewer who opened the PDF beats a regex.

``core_jel_codes`` is the merged view with provenance, and ``jel_codes.csv``
is it on disk. The general-category names come from the AEA's
``classifications.xml`` (the ``X00`` entries, minus their ``: General`` tail).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from citegraph.annotations import load_annotations
from citegraph.io import OutLayout

JEL_GENERAL_CATEGORIES: dict[str, str] = {
    "A": "General Economics and Teaching",
    "B": "History of Economic Thought, Methodology, and Heterodox Approaches",
    "C": "Mathematical and Quantitative Methods",
    "D": "Microeconomics",
    "E": "Macroeconomics and Monetary Economics",
    "F": "International Economics",
    "G": "Financial Economics",
    "H": "Public Economics",
    "I": "Health, Education, and Welfare",
    "J": "Labor and Demographic Economics",
    "K": "Law and Economics",
    "L": "Industrial Organization",
    "M": "Business Administration and Business Economics; Marketing; Accounting; "
    "Personnel Economics",
    "N": "Economic History",
    "O": "Economic Development, Innovation, Technological Change, and Growth",
    "P": "Political Economy and Comparative Economic Systems",
    "Q": "Agricultural and Natural Resource Economics; Environmental and Ecological Economics",
    "R": "Urban, Rural, Regional, Real Estate, and Transportation Economics",
    "Y": "Miscellaneous Categories",
    "Z": "Other Special Topics",
}

# docling leaves font escapes in the text; these are the ones that land in JEL lines.
_ARTEFACTS = {"/uni00A0": " ", "/C1": " ", "/uniFB01": "fi", "/fication": "fication"}

# "JEL", an optional label ("classification", "codes", "classification numbers",
# with the fi-ligature mangled every way docling manages), then the rest of the line.
_LABEL_RE = re.compile(
    r"\bJEL\b[ \t]*[-–:]?[ \t]*"
    r"(?:classi\s*fi?\s*c?ations?|codes?|numbers?)?"
    r"(?:[ \t]+(?:numbers?|codes?))?"
    r"[ \t]*[:.]?[ \t]*(?P<rest>[^\n]*)",
    re.IGNORECASE,
)
_CODE_RE = re.compile(r"\b([A-Z0]) ?(\d{1,2})\b")
_FILLER_RE = re.compile(r"\band\b|[\s,;.·/()\[\]\-–]")
_CODE_FORMAT_RE = re.compile(r"^[A-Z]\d{1,2}$")


def _codes_in(chunk: str) -> list[str]:
    """Codes in ``chunk`` when it is a code list and nothing else."""
    codes = [
        ("O" if letter == "0" else letter) + digits for letter, digits in _CODE_RE.findall(chunk)
    ]
    codes = [c for c in codes if c[0] in JEL_GENERAL_CATEGORIES]
    if not codes or _FILLER_RE.sub("", _CODE_RE.sub("", chunk)).strip():
        return []
    return codes


def extract_jel_codes(markdown: str) -> list[str]:
    """JEL codes printed in ``markdown``, in order, deduplicated.

    Every ``JEL`` block is read — a thesis prints one per chapter — and a
    block's codes are either on the label's own line or in the paragraphs
    right after it, read until the first paragraph that is not a code list
    (an Elsevier "Article info" box prints one code per paragraph). A line
    that mentions JEL and then says anything else is prose, not a code list,
    and contributes nothing.
    """
    text = markdown
    for artefact, replacement in _ARTEFACTS.items():
        text = text.replace(artefact, replacement)
    found: dict[str, None] = {}
    for match in _LABEL_RE.finditer(text):
        rest = match.group("rest").strip()
        if rest:
            found.update(dict.fromkeys(_codes_in(rest)))
            continue
        for paragraph in re.split(r"\n\s*\n", text[match.end() :].lstrip("\n")):
            codes = _codes_in(paragraph)
            if not codes:
                break
            found.update(dict.fromkeys(codes))
    return list(found)


def _parse_manual(value: object, work_id: str) -> list[str]:
    if value is None or pd.isna(value) or not str(value).strip():
        return []
    tokens = [token for token in re.split(r"[;,\s]+", str(value).strip()) if token]
    for token in tokens:
        code = token.upper()
        if not _CODE_FORMAT_RE.match(code) or code[0] not in JEL_GENERAL_CATEGORIES:
            raise ValueError(
                f"work_annotations.csv: work {work_id!r} has JEL_Codes entry {token!r}; "
                "expected a letter and one or two digits, e.g. D71 or Q2"
            )
    return list(dict.fromkeys(token.upper() for token in tokens))


def core_jel_codes(out_dir: str | Path) -> pd.DataFrame:
    """One row per core work: ``JEL_Codes``, ``JEL_General`` and where they came from.

    ``JEL_Source`` is ``manual`` (from ``work_annotations.csv``), ``extracted``
    (from the markdown) or ``missing``. ``JEL_General`` is the sorted set of
    first letters — the level the composition figure plots.
    """
    layout = OutLayout(out_dir)
    works = pd.read_csv(layout.works_csv, index_col="id", keep_default_na=False)
    core = works[works["ring"] == 0]
    annotations = load_annotations(
        layout.work_annotations_csv, works=works, schema_path=layout.annotation_schema_csv
    )
    manual = (
        annotations["JEL_Codes"] if "JEL_Codes" in annotations.columns else pd.Series(dtype=object)
    )

    rows = []
    for work_id, work in core.iterrows():
        codes = _parse_manual(manual.get(work_id), str(work_id))
        source = "manual"
        if not codes:
            markdown = layout.markdown_dir / str(work["source_file"])
            if work["source_file"] and markdown.exists():
                codes = extract_jel_codes(markdown.read_text(encoding="utf-8", errors="replace"))
            source = "extracted" if codes else "missing"
        rows.append(
            {
                "id": work_id,
                "source_file": work["source_file"],
                "Title": work["Title"],
                "JEL_Codes": codes,
                "JEL_General": sorted({code[0] for code in codes}),
                "JEL_Source": source,
            }
        )
    columns = ["id", "source_file", "Title", "JEL_Codes", "JEL_General", "JEL_Source"]
    return pd.DataFrame(rows, columns=columns).set_index("id")


def write_jel_codes(out_dir: str | Path) -> Path:
    """Write ``jel_codes.csv`` (lists joined by ``;``) and return its path."""
    frame = core_jel_codes(out_dir)
    for column in ("JEL_Codes", "JEL_General"):
        frame[column] = frame[column].map(";".join)
    path = OutLayout(out_dir).jel_codes_csv
    frame.to_csv(path)
    return path
