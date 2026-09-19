"""One spreadsheet holding everything the corpus knows about its own authors.

``authors.csv`` describes every author in the corpus — thousands of them,
most encountered once in somebody's bibliography. The question a manuscript
actually asks is narrower: *who wrote my papers, and what do we know about
them?* That answer is spread over half a dozen artifacts, and for paper 4 it
reaches outside ``out_dir`` entirely, into hand-collected career metadata.

This module assembles it into a single workbook keyed on ``author_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from citegraph.io import OutLayout, read_json


@dataclass(frozen=True)
class ExtraSheet:
    """A table from outside ``out_dir``, filtered to the core authors."""

    name: str
    frame: pd.DataFrame
    description: str = ""
    master_columns: tuple[str, ...] = ()
    """Columns lifted onto the ``Authors`` sheet; requires one row per author."""


_GENERATED_SHEETS = ("README", "Authors", "Works", "Review_Flags")


def _validate_extra_sheet_names(extra_sheets: tuple[ExtraSheet, ...]) -> None:
    used = {name.casefold(): name for name in _GENERATED_SHEETS}
    for sheet in extra_sheets:
        folded = sheet.name.casefold()
        if folded in used:
            raise ValueError(
                f"Extra sheet {sheet.name!r} conflicts with workbook sheet {used[folded]!r}"
            )
        used[folded] = sheet.name


def load_extra_sheet(
    name: str,
    path: Path | str,
    *,
    description: str = "",
    master_columns: tuple[str, ...] = (),
) -> ExtraSheet:
    """Read a curated CSV from outside ``out_dir`` as an :class:`ExtraSheet`."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Extra sheet {name!r}: no such file: {path}")
    return ExtraSheet(
        name=name,
        frame=pd.read_csv(path),
        description=description or f"Curated table from {path.name}.",
        master_columns=master_columns,
    )


def _require_openpyxl() -> None:
    """Excel writing is an extra, so the base install stays light."""
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "Writing .xlsx needs openpyxl, which is not a default dependency. "
            'Install it with: pip install "citegraph[excel]"'
        ) from exc


def _author_id_columns(frame: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in frame.columns
        if str(column) == "author_id" or str(column).endswith("_author_id")
    ]


def build_core_author_workbook(
    out_dir: Path | str,
    dest: Path | str | None = None,
    *,
    extra_sheets: tuple[ExtraSheet, ...] = (),
    ring: int = 0,
) -> Path:
    """Write ``core_authors.xlsx`` for the authors of ``ring``'s works."""
    _validate_extra_sheet_names(extra_sheets)
    _require_openpyxl()
    layout = OutLayout(Path(out_dir))
    works = pd.read_csv(layout.works_csv)
    occurrences = pd.read_csv(layout.author_citations_csv)
    authors = pd.read_csv(layout.authors_csv)

    core_works = set(works.loc[works["ring"] == ring, "id"])
    core_ids = set(occurrences.loc[occurrences["record_id"].isin(core_works), "author_id"])
    master = authors[authors["id"].isin(core_ids)].copy()
    master["name_variants"] = master["id"].map(_name_variants(occurrences, core_ids))
    works_sheet = _works_sheet(occurrences, works, core_ids)

    centrality = _optional_frame(layout.author_centrality_csv)
    if centrality is not None and "author_id" in centrality.columns:
        master = master.merge(
            centrality.drop(columns=[c for c in ("Author",) if c in centrality.columns]),
            left_on="id",
            right_on="author_id",
            how="left",
        ).drop(columns=["author_id"])
    review = _review_sheet(layout.author_review_json, core_ids)

    filtered: list[tuple[ExtraSheet, pd.DataFrame, str]] = []
    for sheet in extra_sheets:
        rows = _filter_to_core(sheet, core_ids)
        covered: set[str] = set()
        for column in _author_id_columns(rows):
            covered |= set(rows[column].dropna())
        master[f"in_{sheet.name}"] = master["id"].isin(covered)
        master = _lift_columns(master, sheet, rows)
        n_covered = len(covered & core_ids)
        filtered.append((sheet, rows, f"{n_covered} of {len(core_ids)}"))

    tables: list[tuple[str, pd.DataFrame, str, str, str]] = [
        ("Authors", master, "authors.csv", f"One row per author of a ring-{ring} work.",
         f"{len(master)} of {len(core_ids)}"),
        ("Works", works_sheet, "author_citations.csv + works.csv",
         "One row per (author, work), core and cited alike.", ""),
    ]
    if review is not None:
        tables.append(
            ("Review_Flags", review, "author_review.json",
             "Clusters the author stage flagged as low-confidence.", "")
        )
    tables.extend(
        (sheet.name, rows, "supplied by --extra-csv", sheet.description, coverage)
        for sheet, rows, coverage in filtered
    )

    path = Path(dest) if dest is not None else layout.core_authors_xlsx
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _readme(tables).to_excel(writer, sheet_name="README", index=False)
        for name, frame, _source, _description, _coverage in tables:
            frame.to_excel(writer, sheet_name=name, index=False)
    return path


def _readme(
    tables: list[tuple[str, pd.DataFrame, str, str, str]],
) -> pd.DataFrame:
    """A contents page, so the workbook explains itself to whoever opens it.

    ``Core_authors_covered`` is the column to read first on a curated sheet:
    hand-collected metadata never covers everybody, and the gap is a finding
    about the corpus rather than a fault in the export.
    """
    return pd.DataFrame(
        [
            {
                "Sheet": name,
                "Rows": len(frame),
                "Core_authors_covered": coverage,
                "Source": source,
                "Description": description,
            }
            for name, frame, source, description, coverage in tables
        ]
    )


def _optional_frame(path: Path) -> pd.DataFrame | None:
    """Read a CSV that may not exist, and never fail the export over it."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def _review_sheet(path: Path, core_ids: set[str]) -> pd.DataFrame | None:
    """The author-QC flags, for core authors only."""
    if not path.exists():
        return None
    try:
        flags = read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(flags, list):
        return None
    rows = [
        {
            key: ("; ".join(str(item) for item in value) if isinstance(value, list) else value)
            for key, value in flag.items()
        }
        for flag in flags
        if isinstance(flag, dict) and flag.get("author_id") in core_ids
    ]
    return pd.DataFrame(rows)


def _filter_to_core(sheet: ExtraSheet, core_ids: set[str]) -> pd.DataFrame:
    """Keep a row when *any* of its author-id columns names a core author.

    One rule covers both shapes the curated tables take: a per-person table
    keyed on ``author_id``, and an edge table with ``advisor_author_id`` and
    ``advisee_author_id``, where a tie is worth keeping if either end is
    somebody we published.
    """
    columns = _author_id_columns(sheet.frame)
    if not columns:
        raise ValueError(
            f"Extra sheet {sheet.name!r} has no author-id column; expected "
            f"'author_id' or a column ending in '_author_id', got "
            f"{list(sheet.frame.columns)}"
        )
    keep = pd.Series(False, index=sheet.frame.index)
    for column in columns:
        keep |= sheet.frame[column].isin(core_ids)
    return sheet.frame[keep].copy()


_WORK_COLUMNS = ["ring", "source_file", "Title", "Journal", "Journal_Canonical", "Year"]


def _works_sheet(
    occurrences: pd.DataFrame, works: pd.DataFrame, core_ids: set[str]
) -> pd.DataFrame:
    """Every work a core author appears on, core and cited alike.

    Restricting this to ring 0 would answer a question nobody asked: an
    author is interesting partly *because* the corpus cites their other work.
    """
    mine = occurrences[occurrences["author_id"].isin(core_ids)]
    columns = [column for column in _WORK_COLUMNS if column in works.columns]
    return mine.merge(
        works[["id", *columns]].rename(columns={"id": "record_id"}),
        on="record_id",
        how="left",
    )


def _name_variants(occurrences: pd.DataFrame, core_ids: set[str]) -> dict[str, str]:
    """Every distinct spelling the bibliographies used, per author.

    This is the evidence the cluster was built from, and the fastest way for
    a reader to see that a merge was right — or that it was not.
    """
    mine = occurrences[occurrences["author_id"].isin(core_ids)]
    grouped = mine.groupby("author_id")["raw_author"]
    return {
        author_id: "; ".join(sorted({str(name) for name in names if pd.notna(name)}))
        for author_id, names in grouped
    }


def _lift_columns(
    master: pd.DataFrame, sheet: ExtraSheet, rows: pd.DataFrame
) -> pd.DataFrame:
    """Copy a curated sheet's headline columns onto the master."""
    if not sheet.master_columns:
        return master
    if "author_id" not in rows.columns:
        raise ValueError(
            f"Extra sheet {sheet.name!r} lifts columns into the master but has no "
            "'author_id' column to key them on"
        )
    duplicated = rows["author_id"].duplicated().any()
    if duplicated:
        raise ValueError(
            f"Extra sheet {sheet.name!r} lifts columns into the master but does not "
            "have one row per author; lift from a per-person table instead"
        )
    missing = [name for name in sheet.master_columns if name not in rows.columns]
    if missing:
        raise ValueError(
            f"Extra sheet {sheet.name!r} cannot lift {missing}; it has "
            f"{list(rows.columns)}"
        )
    clash = [name for name in sheet.master_columns if name in master.columns]
    if clash:
        raise ValueError(
            f"Extra sheet {sheet.name!r} would overwrite existing column(s) {clash} "
            "on the Authors sheet; rename them in the source table"
        )
    return master.merge(
        rows[["author_id", *sheet.master_columns]],
        left_on="id",
        right_on="author_id",
        how="left",
    ).drop(columns=["author_id"])
