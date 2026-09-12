"""Tests for the ``citegraph ui`` live monitor.

The server binds a random free port on 127.0.0.1 (loopback only — no real
network) and is exercised with urllib against a corpus synthesized into
``tmp_path``, mirroring the fixtures in ``test_report.py``.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path
from urllib.error import HTTPError

import pytest

from citegraph.html_report import collect_report_data
from citegraph.webui import create_server

_LONG_BODY = ("This paragraph carries enough substantive text to clear the "
              "image-only heuristic. " * 5)


def _write_corpus(out: Path) -> None:
    """Convert + metadata done for two papers; references not run."""
    (out / "markdown").mkdir(parents=True)
    (out / "metadata").mkdir(parents=True)
    (out / "markdown" / "alpha.md").write_text(f"# Alpha\n{_LONG_BODY}", encoding="utf-8")
    (out / "markdown" / "beta.md").write_text(f"# Beta\n{_LONG_BODY}", encoding="utf-8")
    for stem, title in (("alpha", "Alpha Paper"), ("beta", "Beta Paper")):
        (out / "metadata" / f"{stem}.json").write_text(
            json.dumps(
                {"Title": title, "Authors_List": ["Some One"], "Journal": "J", "Year": 2020}
            ),
            encoding="utf-8",
        )
    (out / "sources.csv").write_text(
        "id,Title,Authors,Journal,Year,source_file\n"
        "w-one-2020-alpha,Alpha Paper,Some One,J,2020,alpha.md\n"
        "w-one-2020-beta,Beta Paper,Some One,J,2020,beta.md\n",
        encoding="utf-8",
    )


@pytest.fixture()
def served_corpus(tmp_path: Path):
    out = tmp_path / "out"
    _write_corpus(out)
    server = create_server(out, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], out
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(port: int, path: str) -> tuple[int, str, str]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as resp:
        return resp.status, resp.headers.get("Content-Type", ""), resp.read().decode("utf-8")


# ---------------------------------------------------------------------------
# Page + data endpoints
# ---------------------------------------------------------------------------
def test_index_serves_html(served_corpus) -> None:
    port, _ = served_corpus
    status, ctype, body = _get(port, "/")

    assert status == 200
    assert ctype.startswith("text/html")
    assert "live monitor" in body
    assert "UI_ARCHITECTURE.md" in body


def test_api_data_reflects_corpus(served_corpus) -> None:
    port, out = served_corpus
    status, ctype, body = _get(port, "/api/data")

    assert status == 200
    assert ctype.startswith("application/json")
    data = json.loads(body)
    assert data["out_dir"] == str(out)
    assert {s["label"] for s in data["stages"]} == {
        "convert", "metadata", "references", "dedup", "enrich", "authors",
    }
    assert [p["stem"] for p in data["papers"]] == ["alpha", "beta"]
    # Previews are fetched lazily via /api/markdown, not shipped in the payload.
    assert all(p["markdown_preview"] is None for p in data["papers"])
    assert data["problems"]["total"] >= 0
    assert data["generated_at"]

    recent = data["recent_files"]
    assert recent, "recent_files must list the corpus files"
    assert {"path", "size", "mtime"} <= set(recent[0])
    paths = {f["path"] for f in recent}
    assert "sources.csv" in paths
    # Newest-first ordering.
    mtimes = [f["mtime"] for f in recent]
    assert mtimes == sorted(mtimes, reverse=True)


def test_api_data_works_before_out_dir_exists(tmp_path: Path) -> None:
    server = create_server(tmp_path / "nope", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, body = _get(server.server_address[1], "/api/data")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert status == 200
    data = json.loads(body)
    assert data["out_dir_exists"] is False
    assert data["papers"] == []
    assert data["recent_files"] == []


def test_unknown_path_is_404(served_corpus) -> None:
    port, _ = served_corpus
    with pytest.raises(HTTPError) as excinfo:
        _get(port, "/api/nope")
    with excinfo.value:
        assert excinfo.value.code == 404


# ---------------------------------------------------------------------------
# Markdown endpoint
# ---------------------------------------------------------------------------
def test_api_markdown_serves_preview(served_corpus) -> None:
    port, _ = served_corpus
    status, ctype, body = _get(port, "/api/markdown?stem=alpha")

    assert status == 200
    assert ctype.startswith("text/plain")
    assert body.startswith("# Alpha")
    assert len(body) <= 2_500


def test_api_markdown_unknown_stem_is_404(served_corpus) -> None:
    port, _ = served_corpus
    with pytest.raises(HTTPError) as excinfo:
        _get(port, "/api/markdown?stem=doesnotexist")
    with excinfo.value:
        assert excinfo.value.code == 404


@pytest.mark.parametrize(
    "stem",
    ["../papers", "..%2Fpapers", "/etc/passwd", "a/b", "..", ""],
)
def test_api_markdown_rejects_traversal(served_corpus, stem: str) -> None:
    port, _ = served_corpus
    with pytest.raises(HTTPError) as excinfo:
        _get(port, f"/api/markdown?stem={stem}")
    with excinfo.value:
        assert excinfo.value.code in (400, 404)


# ---------------------------------------------------------------------------
# collect_report_data preview flag
# ---------------------------------------------------------------------------
def test_include_markdown_previews_flag(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _write_corpus(out)

    with_previews = collect_report_data(out)
    without = collect_report_data(out, include_markdown_previews=False)

    assert all(p["markdown_preview"] for p in with_previews["papers"])
    assert all(p["markdown_preview"] is None for p in without["papers"])
    # Char counts and image-only checks are unaffected.
    for a, b in zip(with_previews["papers"], without["papers"], strict=True):
        assert a["markdown"] == b["markdown"]
