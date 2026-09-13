import json

import pandas as pd
import pytest

from citegraph.dedup import DedupConfig
from citegraph.html_report import build_report_html, collect_report_data
from citegraph.io import artifact_fingerprint
from tests.test_reliability_pipeline import setup_cached


def corpus(tmp_path):
    p, md = setup_cached(tmp_path)
    p.dedup_config = DedupConfig(threshold=101)
    sources = p.extract_paper_metadata([md])
    (p.layout.references_dir / "paper.json").write_text(json.dumps([
        dict(Title="Referenced work", Authors_List=["Jane Smith"], Journal="J", Year=2010),
        dict(Title="Referenced work", Authors_List=["Jane Smith"], Journal="J", Year=2010),
    ]))
    refs = p.extract_paper_references([md], sources)
    p.deduplicate(sources, refs)
    p.normalize_authors()
    return p


def test_report_uses_exact_nondefault_decisions(tmp_path, monkeypatch):
    p = corpus(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Report must not recompute history")
    monkeypatch.setattr("citegraph.dedup.canonicalize_works", forbidden)
    monkeypatch.setattr("citegraph.html_report.canonicalize_works", forbidden, raising=False)
    data = collect_report_data(tmp_path)
    assert data["dedup"]["audit_status"] == "current"
    assert data["dedup"]["audit_config"]["threshold"] == 101
    assert data["dedup"]["merge_audit"] == []
    assert data["authors"]["audit_status"] == "current"
    assert "recomputed" not in build_report_html(data)
    audit = json.loads(p.layout.canonicalization_audit_json.read_text())
    assert len(set(audit["citation_cluster_ids"])) == 2
    assert audit["input_fingerprints"] and audit["output_fingerprints"]


@pytest.mark.parametrize("filename", ["sources.csv", "citations_raw.csv", "works.csv", "citation_graph.csv"])
def test_changed_snapshot_marks_work_audit_stale(tmp_path, filename):
    corpus(tmp_path)
    path = tmp_path / filename
    data = pd.read_csv(path)
    data.iloc[0, 0] = "changed"
    data.to_csv(path, index=False)
    panel = collect_report_data(tmp_path)["dedup"]
    assert panel["audit_status"] == "stale"
    assert panel["merge_audit"] == []


def test_missing_audit_is_unavailable_without_recomputation(tmp_path, monkeypatch):
    p = corpus(tmp_path)
    p.layout.canonicalization_audit_json.unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError("No guessed audit")
    monkeypatch.setattr("citegraph.html_report.canonicalize_works", forbidden, raising=False)
    data = collect_report_data(tmp_path)
    assert data["dedup"]["audit_status"] == "unavailable"
    assert data["dedup"]["merge_audit"] == []
    assert not data["errors"]


def test_corrupt_audit_is_a_finding_not_a_report_crash(tmp_path):
    p = corpus(tmp_path)
    p.layout.canonicalization_audit_json.write_text("{broken")
    data = collect_report_data(tmp_path)
    assert data["dedup"]["audit_status"] == "invalid"
    assert any(e["path"].endswith("canonicalization_audit.json") for e in data["errors"])


def test_commented_alias_csv_can_be_fingerprinted(tmp_path):
    path = tmp_path / "author_aliases.csv"
    path.write_text("# note, with a comma\ncluster_id,canonical_id\na,b\n")
    before = artifact_fingerprint(path)
    path.write_text("# changed note, with a comma\ncluster_id,canonical_id\na,b\n")
    assert before != artifact_fingerprint(path)


@pytest.mark.parametrize("filename", ["authors.csv", "author_citations.csv", "author_aliases.csv", "author_overrides.csv"])
def test_changed_author_inputs_or_outputs_mark_audit_stale(tmp_path, filename):
    corpus(tmp_path)
    path = tmp_path / filename
    if path.exists():
        frame = pd.read_csv(path)
        frame.iloc[0, 0] = "changed"
        frame.to_csv(path, index=False)
    else:
        path.write_text("cluster_id,canonical_id\na,b\n")
    assert collect_report_data(tmp_path)["authors"]["audit_status"] == "stale"


def test_saved_work_review_and_author_correction_are_escaped(tmp_path):
    p = corpus(tmp_path)
    p.dedup_config = DedupConfig()
    refs = pd.read_csv(p.layout.citations_raw_csv)
    refs["Title"] = ["Economic policy", "Economic policy without growth"]
    refs.to_csv(p.layout.citations_raw_csv, index=False)
    works, _ = p.deduplicate()
    targets = list(works.index[works.ring == 1])
    pd.DataFrame([dict(action="separate", left_record_id=targets[0], left_position=0,
                       right_record_id=targets[1], right_position=0,
                       reason="<script>alert('test')</script>")]).to_csv(p.layout.author_overrides_csv, index=False)
    p.normalize_authors()
    data = collect_report_data(tmp_path)
    assert data["dedup"]["audit_decisions"][0]["decision"] == "review"
    assert any(row["decision"] == "manual_override" for row in data["authors"]["audit_decisions"])
    html = build_report_html(data)
    assert "negation_conflict" in html
    assert "<script>alert('test')</script>" not in html
    assert "&lt;script&gt;" in html


def test_saved_merged_source_and_citation_observations(tmp_path):
    p = corpus(tmp_path)
    p.dedup_config = DedupConfig()
    p.deduplicate()
    data = collect_report_data(tmp_path)
    assert len(data["dedup"]["merge_audit"]) == 1
    assert len(data["dedup"]["merge_audit"][0]["members"]) == 2
    assert data["authors"]["audit_status"] == "stale"


@pytest.mark.parametrize("field,value", [("decisions", ["bad"]),
                                         ("citation_cluster_ids", ["nonexistent"]),
                                         ("input_fingerprints", {"../../secret": "x" * 64})])
def test_invalid_evidence_shape_is_safe(tmp_path, field, value):
    p = corpus(tmp_path)
    audit = json.loads(p.layout.canonicalization_audit_json.read_text())
    audit[field] = value
    p.layout.canonicalization_audit_json.write_text(json.dumps(audit))
    assert collect_report_data(tmp_path)["dedup"]["audit_status"] == "invalid"
