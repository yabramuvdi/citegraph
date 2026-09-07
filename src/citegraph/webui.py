"""Read-only live-monitoring web UI for a citegraph out directory.

``citegraph ui --out ./out`` serves a single-page dashboard on
``127.0.0.1`` that polls the out directory every couple of seconds and
re-renders as artifacts appear, so a running pipeline can be watched
live. It is the MVP slice of the researcher-facing application proposed
in ``docs/UI_ARCHITECTURE.md``: strictly read-only (it never writes into
``out_dir``), stdlib-only, and built on the same data contract as
``citegraph report`` (:func:`citegraph.html_report.collect_report_data`).

Endpoints:

- ``GET /`` — the page (inline CSS/JS, no external requests).
- ``GET /api/data`` — the report data dict (without markdown previews)
  plus ``recent_files``, the most recently modified files under
  ``out_dir``. Valid even when ``out_dir`` does not exist yet.
- ``GET /api/markdown?stem=<stem>`` — the first ~2,500 chars of one
  converted markdown file, fetched lazily when a paper row is expanded.
  Only bare stems resolving inside ``out_dir/markdown`` are served.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from citegraph.html_report import _MARKDOWN_PREVIEW_CHARS, collect_report_data
from citegraph.io import OutLayout

__all__ = ["create_server"]

_RECENT_FILES_N = 15


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------
def _scan_files(out_dir: Path) -> list[tuple[float, int, Path]]:
    """(mtime, size, path) for every regular file under ``out_dir``."""
    if not out_dir.is_dir():
        return []
    entries: list[tuple[float, int, Path]] = []
    for path in out_dir.rglob("*"):
        try:
            if not path.is_file():
                continue
            stat = path.stat()
        except OSError:  # file vanished mid-scan — a live run is writing
            continue
        entries.append((stat.st_mtime, stat.st_size, path))
    return entries


def _recent_files(entries: list[tuple[float, int, Path]], out_dir: Path) -> list[dict]:
    newest = sorted(entries, key=lambda e: e[0], reverse=True)[:_RECENT_FILES_N]
    return [
        {
            "path": str(path.relative_to(out_dir)),
            "size": size,
            "mtime": datetime.fromtimestamp(mtime, UTC).isoformat(timespec="seconds"),
        }
        for mtime, size, path in newest
    ]


def _build_payload_text(out_dir: Path) -> tuple[str, tuple]:
    """Serialized ``/api/data`` payload plus the directory fingerprint it reflects."""
    entries = _scan_files(out_dir)
    fingerprint = (len(entries), max((e[0] for e in entries), default=0.0))
    data = collect_report_data(out_dir, include_markdown_previews=False)
    data["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    data["recent_files"] = _recent_files(entries, out_dir)
    return json.dumps(data), fingerprint


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
class _UIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], out_dir: Path, verbose: bool) -> None:
        super().__init__(address, _Handler)
        self.out_dir = Path(out_dir)
        self.verbose = verbose
        # /api/data is recollected only when the directory fingerprint
        # (file count + newest mtime) changes; polls in between reuse the
        # cached text, which also lets the client skip re-rendering.
        self._payload_lock = threading.Lock()
        self._payload_fingerprint: tuple | None = None
        self._payload_text = ""

    def payload_text(self) -> str:
        with self._payload_lock:
            entries = _scan_files(self.out_dir)
            fingerprint = (len(entries), max((e[0] for e in entries), default=0.0))
            if fingerprint != self._payload_fingerprint:
                self._payload_text, self._payload_fingerprint = _build_payload_text(
                    self.out_dir
                )
            return self._payload_text


class _Handler(BaseHTTPRequestHandler):
    server: _UIServer
    server_version = "citegraph-ui"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        if self.server.verbose:
            super().log_message(format, *args)

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, code: int, text: str) -> None:
        self._send(code, "text/plain; charset=utf-8", text.encode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, "text/html; charset=utf-8", _PAGE.encode("utf-8"))
        elif parsed.path == "/api/data":
            self._send(
                200,
                "application/json; charset=utf-8",
                self.server.payload_text().encode("utf-8"),
            )
        elif parsed.path == "/api/markdown":
            self._handle_markdown(parsed.query)
        else:
            self._send_text(404, "not found")

    def _handle_markdown(self, query: str) -> None:
        stem = (parse_qs(query).get("stem") or [""])[0]
        if not stem or "/" in stem or "\\" in stem or ".." in stem:
            self._send_text(400, "invalid stem")
            return
        markdown_dir = OutLayout(self.server.out_dir).markdown_dir.resolve()
        candidate = (markdown_dir / f"{stem}.md").resolve()
        if not candidate.is_relative_to(markdown_dir):
            self._send_text(400, "invalid stem")
            return
        if not candidate.is_file():
            self._send_text(404, "unknown stem")
            return
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            self._send_text(404, "unreadable markdown")
            return
        self._send_text(200, text[:_MARKDOWN_PREVIEW_CHARS])


def create_server(out_dir: Path, *, port: int = 8765, verbose: bool = False) -> _UIServer:
    """Bind the live-monitor server on ``127.0.0.1:port`` (0 = any free port).

    The caller drives it: ``server.serve_forever()`` /
    ``server.shutdown()`` / ``server.server_close()``. The server never
    writes into ``out_dir`` and works before the directory exists.
    """
    return _UIServer(("127.0.0.1", port), Path(out_dir), verbose)


# ---------------------------------------------------------------------------
# Frontend (static; all state comes from /api/data)
# ---------------------------------------------------------------------------
_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>citegraph — live monitor</title>
<style>
:root {
  --paper: #f6f7f4; --surface: #fdfdfb; --line: #dde2db;
  --ink: #1b2420; --ink-soft: #55605a;
  --accent: #1f6a4e; --accent-soft: #e3eee8;
  --brass: #a07a2f; --warn: #b4552d; --warn-soft: #f6e9e2;
  --code-bg: #eef1ec;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #131a17; --surface: #1a231f; --line: #2c3833;
    --ink: #e6eae4; --ink-soft: #9aa8a0;
    --accent: #58b78e; --accent-soft: #1e3a2f;
    --brass: #c9a45c; --warn: #d98a63; --warn-soft: #3a2820;
    --code-bg: #101613;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--paper); color: var(--ink);
  font-family: system-ui, sans-serif; font-size: 15px; line-height: 1.5;
}
h1, h2, h3 { font-family: "Iowan Old Style", Georgia, serif; line-height: 1.2; }
h1 { font-size: 1.6rem; margin: 0; }
h2 { font-size: 1.25rem; margin: 0 0 0.75rem; }
code, .mono, pre { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.86em; }
code { background: var(--code-bg); padding: 0.1em 0.35em; border-radius: 4px; }
main { max-width: 1180px; margin: 0 auto; padding: 1rem 1.25rem 4rem; }
section {
  background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  padding: 1.1rem 1.25rem 1.35rem; margin-top: 1.25rem;
}
.eyebrow {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.68rem;
  color: var(--brass); margin: 0 0 0.35rem;
}
.soft { color: var(--ink-soft); }
header.page { padding: 1.4rem 0 0; }
.liveline {
  display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap;
  color: var(--ink-soft); font-size: 0.85rem; margin-top: 0.35rem;
}
.dot {
  width: 9px; height: 9px; border-radius: 50%; background: var(--accent);
  display: inline-block; animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 50% { opacity: 0.3; } }
@media (prefers-reduced-motion: reduce) { .dot { animation: none; } }
body.stale .dot { background: var(--warn); animation: none; }
body.paused .dot { background: var(--ink-soft); animation: none; }
#pause {
  font: inherit; font-size: 0.8rem; color: var(--ink); background: var(--surface);
  border: 1px solid var(--line); border-radius: 6px; padding: 0.15rem 0.6rem; cursor: pointer;
}
#pause:hover { border-color: var(--accent); color: var(--accent); }
#stale-note { color: var(--warn); display: none; }
body.stale #stale-note { display: inline; }
.pill {
  display: inline-block; padding: 0.05em 0.55em; border-radius: 999px;
  font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.72rem;
  white-space: nowrap; vertical-align: middle;
}
.pill-ok { background: var(--accent-soft); color: var(--accent); }
.pill-warn { background: var(--warn-soft); color: var(--warn); }
.pill-muted { background: var(--line); color: var(--ink-soft); }
.stage-strip {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.6rem; margin-top: 0.9rem;
}
.stage { border: 1px solid var(--line); border-radius: 8px; padding: 0.6rem 0.7rem;
  background: var(--surface); }
.stage-done { border-color: var(--accent); }
.stage-name {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.68rem;
  color: var(--ink-soft); margin-bottom: 0.25rem;
}
.stage-detail { font-size: 0.8rem; color: var(--ink-soft); margin-top: 0.3rem; }
.stage-hint { margin-top: 0.3rem; font-size: 0.72rem; overflow-x: auto; }
.problem-cards {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 0.6rem;
}
.problem-card {
  display: flex; align-items: baseline; gap: 0.5rem;
  border: 1px solid var(--warn); border-radius: 8px; padding: 0.55rem 0.7rem;
  background: var(--warn-soft); color: var(--ink);
}
.problem-count {
  font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 1.15rem;
  font-variant-numeric: tabular-nums; color: var(--warn); font-weight: 600;
}
.problem-label { font-size: 0.82rem; }
.all-clear { color: var(--accent); margin: 0; }
.placeholder { color: var(--ink-soft); margin: 0; }
.controls { display: flex; gap: 1rem; align-items: center; margin-bottom: 0.7rem; flex-wrap: wrap; }
.controls input[type="search"] {
  background: var(--paper); color: var(--ink); border: 1px solid var(--line);
  border-radius: 6px; padding: 0.35rem 0.6rem; min-width: 240px; font: inherit;
}
.controls label { font-size: 0.85rem; color: var(--ink-soft); }
.table-scroll { overflow-x: auto; }
.paper-table { min-width: 900px; border-top: 1px solid var(--line); }
.paper-row { border-bottom: 1px solid var(--line); }
.paper-row summary, .paper-head {
  display: grid; grid-template-columns: 2.2fr 1.2fr 2.6fr 1.1fr 1.6fr;
  gap: 0.7rem; padding: 0.45rem 0.3rem; align-items: baseline; cursor: pointer;
  list-style: none;
}
.paper-head { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--ink-soft); font-family: ui-monospace, "SF Mono", Menlo, monospace;
  cursor: default; }
.paper-row summary::-webkit-details-marker { display: none; }
.paper-row summary:hover { background: var(--code-bg); }
.col { overflow: hidden; text-overflow: ellipsis; }
.col .mono { font-variant-numeric: tabular-nums; }
.col-notes { font-size: 0.8rem; color: var(--warn); }
.cell-title { font-size: 0.88rem; }
.paper-body { padding: 0.5rem 0.75rem 1rem; background: var(--code-bg); border-radius: 6px;
  margin: 0 0.3rem 0.7rem; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 0.15rem 1rem; margin: 0.5rem 0; }
.kv dt { color: var(--ink-soft); font-size: 0.8rem; }
.kv dd { margin: 0; }
.md-preview {
  max-height: 260px; overflow: auto; background: var(--paper);
  border: 1px solid var(--line); border-radius: 6px; padding: 0.6rem;
  white-space: pre-wrap; word-break: break-word; margin: 0.25rem 0;
}
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
th, td { text-align: left; padding: 0.35rem 0.6rem; border-bottom: 1px solid var(--line);
  vertical-align: top; }
th { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--ink-soft); font-family: ui-monospace, "SF Mono", Menlo, monospace; }
td.num { font-variant-numeric: tabular-nums; text-align: right;
  font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.82rem; }
footer { margin-top: 2rem; color: var(--ink-soft); font-size: 0.8rem; }
</style></head><body>
<main>
  <header class="page">
    <p class="eyebrow">citegraph · live monitor</p>
    <h1 id="title">out</h1>
    <div class="liveline">
      <span class="dot" id="dot"></span>
      <span class="mono" id="outdir"></span>
      <span id="ago">connecting…</span>
      <span id="stale-note">server stopped? retrying…</span>
      <button id="pause" type="button">pause</button>
    </div>
    <div class="stage-strip" id="stages"></div>
  </header>
  <section>
    <p class="eyebrow">summary</p>
    <h2>Problems</h2>
    <div id="problems"></div>
  </section>
  <section>
    <p class="eyebrow">activity</p>
    <h2>Recent files</h2>
    <div id="recent" class="table-scroll"></div>
  </section>
  <section>
    <p class="eyebrow">per-paper</p>
    <h2>Papers</h2>
    <div class="controls">
      <input id="filter-text" type="search" placeholder="filter by stem or title…">
      <label><input id="filter-problems" type="checkbox"> problems only</label>
      <span id="paper-count" class="soft"></span>
    </div>
    <div id="papers"></div>
  </section>
  <footer>read-only MVP — see <code>docs/UI_ARCHITECTURE.md</code> for the full
    application plan.</footer>
</main>
<script>
(function () {
  'use strict';
  var POLL_MS = 2000;
  var paused = false;
  var lastText = null;
  var lastFetchedAt = null;
  var openStems = {};      // stem -> true while its row is expanded
  var mdCache = {};        // stem -> fetched markdown preview (or pending marker)

  var PILL = {
    ok: 'ok', done: 'ok',
    zero: 'warn', failed: 'warn', corrupt: 'warn', error: 'warn',
    'image-only': 'warn', missing: 'warn', partial: 'warn',
    pending: 'muted', skipped: 'muted', not_run: 'muted', optional_not_run: 'muted'
  };

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }
  function pill(label, status) {
    return el('span', 'pill pill-' + (PILL[status] || 'muted'), label);
  }
  function yearStr(y) { return y ? String(y) : '?'; }

  function renderStages(stages) {
    var strip = document.getElementById('stages');
    strip.textContent = '';
    stages.forEach(function (s) {
      var cell = el('div', 'stage' + (s.status === 'done' ? ' stage-done' : ''));
      cell.appendChild(el('div', 'stage-name', s.label));
      var mark = { done: '\\u2713 done', partial: 'partial', not_run: 'not run' }[s.status]
        || 'optional';
      cell.appendChild(pill(mark, s.status));
      cell.appendChild(el('div', 'stage-detail', s.detail));
      if (s.hint) {
        var hint = el('div', 'stage-hint');
        hint.appendChild(el('code', null, s.hint));
        cell.appendChild(hint);
      }
      strip.appendChild(cell);
    });
  }

  function renderProblems(problems) {
    var box = document.getElementById('problems');
    box.textContent = '';
    var cards = (problems && problems.cards) || [];
    if (!cards.length) {
      box.appendChild(el('p', 'all-clear',
        'No problems detected at the stages run so far.'));
      return;
    }
    var grid = el('div', 'problem-cards');
    cards.forEach(function (c) {
      var card = el('div', 'problem-card');
      card.appendChild(el('span', 'problem-count', c.count));
      card.appendChild(el('span', 'problem-label', c.label));
      grid.appendChild(card);
    });
    box.appendChild(grid);
  }

  function renderRecent(files) {
    var box = document.getElementById('recent');
    box.textContent = '';
    if (!files || !files.length) {
      box.appendChild(el('p', 'placeholder', 'No files yet.'));
      return;
    }
    var table = el('table');
    var thead = el('thead'); var hr = el('tr');
    ['file', 'size', 'modified (UTC)'].forEach(function (h) {
      hr.appendChild(el('th', null, h));
    });
    thead.appendChild(hr); table.appendChild(thead);
    var tbody = el('tbody');
    files.forEach(function (f) {
      var tr = el('tr');
      var pathCell = el('td'); pathCell.appendChild(el('span', 'mono', f.path));
      tr.appendChild(pathCell);
      tr.appendChild(el('td', 'num', f.size.toLocaleString()));
      tr.appendChild(el('td', 'num', f.mtime.replace('T', ' ').replace('+00:00', '')));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    box.appendChild(table);
  }

  function loadMarkdown(stem, pre) {
    if (mdCache[stem] !== undefined) { pre.textContent = mdCache[stem]; return; }
    mdCache[stem] = 'loading\\u2026';
    fetch('/api/markdown?stem=' + encodeURIComponent(stem))
      .then(function (r) { return r.ok ? r.text() : '(no markdown preview)'; })
      .catch(function () { return '(no markdown preview)'; })
      .then(function (text) {
        mdCache[stem] = text;
        pre.textContent = text;
      });
    pre.textContent = mdCache[stem];
  }

  function paperBody(p) {
    var body = el('div', 'paper-body');
    if (p.metadata.status === 'ok') {
      var kv = el('dl', 'kv');
      [['Title', p.metadata.title], ['Authors', p.metadata.authors],
       ['Journal', p.metadata.journal], ['Year', yearStr(p.metadata.year)]]
        .forEach(function (pair) {
          kv.appendChild(el('dt', null, pair[0]));
          kv.appendChild(el('dd', null, pair[1]));
        });
      body.appendChild(kv);
    }
    if (p.refs_preview && p.refs_preview.length) {
      body.appendChild(el('h3', null,
        'Extracted references (first ' + p.refs_preview.length + ')'));
      var ol = el('ol');
      p.refs_preview.forEach(function (r) {
        var li = el('li', null, r.title + ' ');
        li.appendChild(el('span', 'soft',
          '\\u2014 ' + r.authors + ' (' + yearStr(r.year) + ')'));
        ol.appendChild(li);
      });
      if (p.refs_more) { ol.appendChild(el('li', 'soft', '\\u2026and ' + p.refs_more + ' more')); }
      body.appendChild(ol);
    }
    if (p.markdown.status === 'ok' || p.markdown.status === 'image-only') {
      body.appendChild(el('h3', null, 'Markdown preview'));
      var pre = el('pre', 'md-preview', '');
      body.appendChild(pre);
      loadMarkdown(p.stem, pre);
    }
    if (!body.childNodes.length) {
      body.appendChild(el('p', 'soft', 'Nothing cached for this paper yet.'));
    }
    return body;
  }

  function paperRow(p) {
    var details = el('details', 'paper-row');
    details.dataset.stem = p.stem;
    details.dataset.title = p.metadata.title || '';
    details.dataset.problem = p.problem ? '1' : '0';
    var summary = el('summary');
    var stemCol = el('span', 'col mono', p.stem);
    var mdCol = el('span', 'col');
    if (p.markdown.chars !== null && p.markdown.chars !== undefined) {
      mdCol.appendChild(el('span', 'mono', p.markdown.chars.toLocaleString() + ' '));
    }
    mdCol.appendChild(pill(p.markdown.status, p.markdown.status));
    var metaCol = el('span', 'col');
    metaCol.appendChild(pill(p.metadata.status, p.metadata.status));
    if (p.metadata.status === 'ok') {
      var t = el('span', 'cell-title', ' ' + p.metadata.title + ' ');
      t.appendChild(el('span', 'soft', '(' + yearStr(p.metadata.year) + ')'));
      metaCol.appendChild(t);
    }
    var refsCol = el('span', 'col');
    if (p.references.count !== null && p.references.count !== undefined) {
      refsCol.appendChild(el('span', 'mono', p.references.count + ' '));
    }
    refsCol.appendChild(pill(p.references.status, p.references.status));
    var notesCol = el('span', 'col col-notes', (p.notes || []).join('; '));
    [stemCol, mdCol, metaCol, refsCol, notesCol].forEach(function (c) {
      summary.appendChild(c);
    });
    details.appendChild(summary);
    if (openStems[p.stem]) {
      details.open = true;
      details.appendChild(paperBody(p));
    }
    details.addEventListener('toggle', function () {
      openStems[p.stem] = details.open;
      if (details.open && details.childNodes.length === 1) {
        details.appendChild(paperBody(p));
      }
    });
    return details;
  }

  function renderPapers(papers) {
    var box = document.getElementById('papers');
    box.textContent = '';
    if (!papers.length) {
      box.appendChild(el('p', 'placeholder',
        'Nothing processed yet \\u2014 waiting for citegraph convert output\\u2026'));
      applyFilter();
      return;
    }
    var scroll = el('div', 'table-scroll');
    var table = el('div', 'paper-table');
    var head = el('div', 'paper-row paper-head');
    ['stem', 'conversion', 'metadata', 'references', 'notes'].forEach(function (h) {
      head.appendChild(el('span', 'col', h));
    });
    table.appendChild(head);
    papers.forEach(function (p) { table.appendChild(paperRow(p)); });
    scroll.appendChild(table);
    box.appendChild(scroll);
    applyFilter();
  }

  var q = document.getElementById('filter-text');
  var cb = document.getElementById('filter-problems');
  function applyFilter() {
    var needle = q.value.trim().toLowerCase();
    var rows = document.querySelectorAll('details.paper-row');
    var shown = 0;
    rows.forEach(function (row) {
      var hay = (row.dataset.stem + ' ' + row.dataset.title).toLowerCase();
      var ok = (!needle || hay.indexOf(needle) !== -1) &&
               (!cb.checked || row.dataset.problem === '1');
      row.hidden = !ok;
      if (ok) { shown += 1; }
    });
    document.getElementById('paper-count').textContent =
      'showing ' + shown + ' of ' + rows.length;
  }
  q.addEventListener('input', applyFilter);
  cb.addEventListener('change', applyFilter);

  function render(data) {
    document.getElementById('title').textContent =
      data.out_dir.split('/').filter(Boolean).pop() || 'out';
    document.getElementById('outdir').textContent = data.out_dir;
    renderStages(data.stages || []);
    renderProblems(data.problems);
    renderRecent(data.recent_files);
    renderPapers(data.papers || []);
  }

  function tick() {
    if (paused) { return; }
    fetch('/api/data')
      .then(function (r) { return r.text(); })
      .then(function (text) {
        document.body.classList.remove('stale');
        lastFetchedAt = Date.now();
        if (text !== lastText) {
          lastText = text;
          render(JSON.parse(text));
        }
      })
      .catch(function () { document.body.classList.add('stale'); });
  }

  function updateAgo() {
    var ago = document.getElementById('ago');
    if (paused) { ago.textContent = 'paused'; return; }
    if (lastFetchedAt === null) { return; }
    ago.textContent = 'updated ' + Math.round((Date.now() - lastFetchedAt) / 1000) + 's ago';
  }

  document.getElementById('pause').addEventListener('click', function () {
    paused = !paused;
    document.body.classList.toggle('paused', paused);
    this.textContent = paused ? 'resume' : 'pause';
    updateAgo();
    if (!paused) { tick(); }
  });

  setInterval(tick, POLL_MS);
  setInterval(updateAgo, 1000);
  tick();
})();
</script>
</body></html>
"""
