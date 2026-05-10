"""Offline evaluation harness for citegraph.

Compares pipeline outputs under ``out/`` against hand-labelled gold files
under ``evaluation/gold/``. Reports:

- Metadata field-level accuracy (exact for Year, fuzzy for text fields).
- Reference recall / precision (fuzzy title matching).
- Optional deduplication F1 if ``gold/dedup_pairs.json`` is present.

Run::

    python evaluation/evaluate.py --gold evaluation/gold --out ./out
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz.fuzz import ratio

FUZZY_TITLE_THRESHOLD = 90.0
FUZZY_JOURNAL_THRESHOLD = 90.0


def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def _author_set(authors: list[str] | str | None) -> set[str]:
    if authors is None:
        return set()
    if isinstance(authors, str):
        # Round-trip through CSV turns lists into their repr; recover.
        s = authors.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    authors = parsed
                else:
                    authors = [s]
            except (SyntaxError, ValueError):
                authors = [s]
        else:
            authors = [a.strip() for a in s.replace(";", ",").split(",")]
    surnames: set[str] = set()
    for a in authors:
        a = a.strip()
        if not a:
            continue
        if "," in a:
            surnames.add(a.split(",")[0].strip().lower())
        else:
            tokens = [t for t in a.split() if t]
            if tokens:
                surnames.add(tokens[-1].lower())
    return surnames


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class MetadataReport:
    n_papers: int = 0
    title_exact: int = 0
    title_fuzzy: int = 0
    year_exact: int = 0
    journal_fuzzy: int = 0
    author_jaccard_sum: float = 0.0

    def add(self, gold: dict, pred: dict) -> None:
        self.n_papers += 1
        if _norm(gold["Title"]) == _norm(pred.get("Title")):
            self.title_exact += 1
        if ratio(_norm(gold["Title"]), _norm(pred.get("Title"))) >= FUZZY_TITLE_THRESHOLD:
            self.title_fuzzy += 1
        try:
            if int(gold["Year"]) == int(pred.get("Year")):
                self.year_exact += 1
        except (TypeError, ValueError):
            pass
        if (
            ratio(_norm(gold.get("Journal")), _norm(pred.get("Journal")))
            >= FUZZY_JOURNAL_THRESHOLD
        ):
            self.journal_fuzzy += 1
        self.author_jaccard_sum += _jaccard(
            _author_set(gold.get("Authors_List") or gold.get("Authors")),
            _author_set(pred.get("Authors_List") or pred.get("Authors")),
        )

    def to_dict(self) -> dict:
        n = max(self.n_papers, 1)
        return {
            "n_papers": self.n_papers,
            "title_exact_acc": self.title_exact / n,
            "title_fuzzy_acc": self.title_fuzzy / n,
            "year_exact_acc": self.year_exact / n,
            "journal_fuzzy_acc": self.journal_fuzzy / n,
            "authors_avg_jaccard": self.author_jaccard_sum / n,
        }


@dataclass
class ReferenceReport:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    per_paper: list[dict] = field(default_factory=list)

    def add(self, gold_refs: list[dict], pred_refs: list[dict], paper_id: str) -> None:
        gold_titles = [_norm(r.get("Title")) for r in gold_refs]
        pred_titles = [_norm(r.get("Title")) for r in pred_refs]

        matched_pred: set[int] = set()
        tp = 0
        for gt in gold_titles:
            best_idx, best_score = -1, 0.0
            for j, pt in enumerate(pred_titles):
                if j in matched_pred:
                    continue
                s = ratio(gt, pt)
                if s > best_score:
                    best_idx, best_score = j, s
            if best_score >= FUZZY_TITLE_THRESHOLD and best_idx != -1:
                matched_pred.add(best_idx)
                tp += 1

        fn = len(gold_titles) - tp
        fp = len(pred_titles) - tp
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.per_paper.append(
            {
                "paper_id": paper_id,
                "n_gold": len(gold_titles),
                "n_pred": len(pred_titles),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    def to_dict(self) -> dict:
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "per_paper": self.per_paper,
        }


def _eval_dedup(out_dir: Path, gold_dir: Path) -> dict | None:
    pairs_path = gold_dir / "dedup_pairs.json"
    refs_csv = out_dir / "references_raw.csv"
    refs_dedup = out_dir / "references.csv"
    graph_csv = out_dir / "citation_graph.csv"
    if not pairs_path.exists() or not refs_csv.exists() or not refs_dedup.exists():
        return None

    pairs = json.loads(pairs_path.read_text())
    raw = pd.read_csv(refs_csv)
    graph = pd.read_csv(graph_csv) if graph_csv.exists() else pd.DataFrame()
    if "cited_id" not in raw.columns and not graph.empty:
        # In the current pipeline the cited_id ends up only in the graph;
        # we can recover the (raw_index -> cluster_id) mapping by re-running
        # dedup with the same config. Skip in that case.
        return {"note": "dedup_pairs eval skipped: raw refs lack cited_id column"}

    correct = 0
    total = len(pairs)
    for left_idx, right_idx in pairs:
        if left_idx >= len(raw) or right_idx >= len(raw):
            continue
        if raw.iloc[left_idx].get("cited_id") == raw.iloc[right_idx].get("cited_id"):
            correct += 1
    return {"n_pairs": total, "n_correct": correct, "accuracy": correct / max(total, 1)}


def evaluate(gold_dir: Path, out_dir: Path) -> dict:
    papers_csv = out_dir / "papers.csv"
    if not papers_csv.exists():
        raise FileNotFoundError(f"{papers_csv} not found; run the pipeline first.")
    papers = pd.read_csv(papers_csv).set_index("source_file")

    references_dir = out_dir / "references"
    if not references_dir.exists():
        raise FileNotFoundError(
            f"{references_dir} not found; the pipeline writes per-paper reference JSONs there."
        )

    metadata_report = MetadataReport()
    reference_report = ReferenceReport()

    for gold_file in sorted(gold_dir.glob("*.json")):
        if gold_file.name == "dedup_pairs.json":
            continue
        gold = json.loads(gold_file.read_text())
        source_file = gold["source_file"]
        if source_file not in papers.index:
            print(f"[skip] {source_file} not in papers.csv")
            continue
        pred_meta = papers.loc[source_file].to_dict()
        metadata_report.add(gold["metadata"], pred_meta)

        ref_cache = references_dir / Path(source_file).with_suffix(".json").name
        if not ref_cache.exists():
            print(f"[skip] no extracted references for {source_file}")
            continue
        pred_refs = json.loads(ref_cache.read_text())
        reference_report.add(gold["references"], pred_refs, paper_id=source_file)

    return {
        "metadata": metadata_report.to_dict(),
        "references": reference_report.to_dict(),
        "dedup": _eval_dedup(out_dir, gold_dir),
    }


def _format_report(report: dict) -> str:
    lines = ["# Evaluation results", ""]
    md = report["metadata"]
    lines.append(f"## Metadata ({md['n_papers']} papers)")
    lines.append("")
    lines.append(f"- Title (exact): {md['title_exact_acc']:.1%}")
    lines.append(f"- Title (fuzzy >= {FUZZY_TITLE_THRESHOLD:.0f}): {md['title_fuzzy_acc']:.1%}")
    lines.append(f"- Year (exact): {md['year_exact_acc']:.1%}")
    lines.append(f"- Journal (fuzzy >= {FUZZY_JOURNAL_THRESHOLD:.0f}): {md['journal_fuzzy_acc']:.1%}")
    lines.append(f"- Authors (avg Jaccard of surname sets): {md['authors_avg_jaccard']:.2f}")

    refs = report["references"]
    lines.append("")
    lines.append("## References")
    lines.append("")
    lines.append(f"- TP: {refs['tp']}, FP: {refs['fp']}, FN: {refs['fn']}")
    lines.append(f"- Precision: {refs['precision']:.1%}")
    lines.append(f"- Recall:    {refs['recall']:.1%}")
    lines.append(f"- F1:        {refs['f1']:.1%}")

    dedup = report.get("dedup")
    if dedup:
        lines.append("")
        lines.append("## Dedup")
        lines.append("")
        if "note" in dedup:
            lines.append(f"- {dedup['note']}")
        else:
            lines.append(
                f"- {dedup['n_correct']}/{dedup['n_pairs']} duplicate pairs clustered "
                f"correctly ({dedup['accuracy']:.1%})"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=Path("evaluation/gold"))
    parser.add_argument("--out", type=Path, default=Path("./out"))
    parser.add_argument(
        "--results-md", type=Path, default=Path("evaluation/RESULTS.md"),
    )
    args = parser.parse_args()

    report = evaluate(args.gold, args.out)
    text = _format_report(report)
    print(text)
    args.results_md.parent.mkdir(parents=True, exist_ok=True)
    args.results_md.write_text(text, encoding="utf-8")
    json_path = args.results_md.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.results_md} and {json_path}")


if __name__ == "__main__":
    main()
