---
name: styling-econ-figures
description: Use when creating, restyling, or reviewing any matplotlib/seaborn figure in this repo that is headed for an economics manuscript, working paper, replication notebook, or journal submission (AER, QJE, JPE, Econometrica, REStud) — including notebooks under examples/ and anything saved to a figures/ directory as PDF or PNG.
---

# Styling Economics-Journal Figures

## Overview

Top economics journals print figures that are quiet, legible in grayscale, and
explained by the caption rather than by text inside the plot.
`citegraph.plotting` encodes those conventions; this skill says how to apply
them. Core principle: **the figure shows the data, the caption explains it.**

Sourced conventions and the rationale for each rule: `references/conventions.md`.

## When to use

- Any plot in `examples/*.ipynb`, a "paper replication" section, or code that
  saves to `figures/`.
- The user says journal, manuscript, paper, publication-quality, submission,
  AER/QJE style, or "make the plots consistent".
- **Not** for the HTML QC dashboard (`html_report.py`, `webui.py`): screen
  charts follow the `dataviz` skill instead.

## Procedure

1. **Form before style.** Counts or rankings → bar (horizontal when labels are
   names); time series → line, or bar when years are few; part-to-whole →
   stacked bar; more than four series → small multiples. Never dual axes, pies,
   3D, or stat tiles / hero numbers.
2. **Open the style.** `with econ_style():` (grayscale default). Use
   `palette="color"` only when more than three series must be told apart, and
   keep the dashes/hatches so a grayscale print still works.
3. **Size at the final printed width.** `figure_size("text")` is 6.5 in for a
   US-letter manuscript; `"journal"` 4.75 in; `"column"` 3.25 in. Two panels
   side by side: `ratio` 0.42–0.5. Twenty horizontal bars: `ratio` ≈ 0.8.
   Never draw an 11–13 in canvas and let the journal shrink it.
4. **One plot per file.** Save each plot as its own figure so the author can
   arrange and number them in the manuscript; combine panels only when the user
   asks for a side-by-side comparison. If panels are combined, head them with
   `panel_title(ax, "A", "Annual counts")`. Never a `suptitle`, subtitle line,
   or source/notes text inside the figure. Leave `number` unset in captions
   unless the figure order is final.
5. **Annual series by default.** Plot time at the yearly resolution; bin into
   five-year periods only when the user asks. If a share is plotted per year,
   the notes say denominators are small and point to the companion counts figure.
6. **Marks.** Bars: call `bar(ax, …)` / `barh(ax, …)`, **never `ax.bar`**.
   Matplotlib takes an unspecified bar colour from the property cycle, whose
   first entry is ink, so a bare `ax.bar` comes out solid black no matter what
   `patch.facecolor` says — no rcParam can express "black lines, gray bars".
   The helpers apply `BAR_FILL` with a black edge and `BAR_WIDTH`; an explicit
   `color=` still wins, for stacked or multi-series bars. Baseline at zero,
   `integer_ticks` / `percent_ticks`. Two or more series: `GRAYS` + `HATCHES`
   and a frameless legend. `value_labels` only where the exact number matters
   (a short ranking, five-year totals); never on a dense annual series, never
   bold, never inside a bar.
7. **Benchmarks** as `reference_line(ax, value, label="All periods: 66%")` —
   dashed black hairline with a small ink label at its right end. If any bar or
   value label sits near the line at the right edge, pass `in_legend=True` and
   call `ax.legend()` instead of nudging text or adding white halos.
8. **Tick labels stay horizontal.** Dense years: `year_ticks(ax, first, last)`
   thins to a tick every 4–5 years and pads the limits. Five-year periods:
   two-line labels (`"1995–\n1999"`) or abbreviated (`"1995–99"`). Rotation is
   the last resort, not the first.

8b. **Networks.** `network_axes(ax)` strips the chrome; `draw_node` carries
   membership in its *fill* and nothing else — `NODE_FILL` for a node inside the
   population described, `NODE_OPEN` for one outside it, `focus=True` for the
   single node a figure is built around. Connectors are `hairline` /
   `arrow_props`, band separators `separator_rule`. Never bold a focus label:
   the fill already distinguishes it.

   Where connectors can touch, `junction_dot` marks the points where they
   **meet** — which is what lets a bare crossing mean they don't. Any router
   drawing a graph that is not a forest needs this: once a node can have two
   parents, one connector has to run past rows another is using, and a
   T-junction reads exactly like an X-crossing. Left unmarked it invents ties —
   a doctoral-lineage figure here showed two people as advisor and student who
   share no edge at all. Dot only points with three or more directions leaving
   them; a dot on every corner trains the reader to ignore dots.

8c. **Boxed nodes and flows.** A node that has to carry a second fact — a name
   over an institution — is `box_node`, and it fills with **`BOX_FILL`, never
   `NODE_FILL`**: a box is orders of magnitude more area than a marker, and at
   `NODE_FILL` the subtitle gray *equals* the fill, so every second line inside a
   filled box disappears. Match the key to the mark with
   `node_legend(..., shape="box")`. Where the second line means one thing on one
   fill and another on the other, say so **in the legend**, beside the fill that
   signals it — a caption is too far away to stop the misreading. `flow_band`
   draws one ribbon of a two-column flow; classify stayers from the *ungrouped*
   columns, because once a tail is pooled "Other" on the left and "Other" on the
   right are different sets and a mover gets counted as a stayer.
9. **Save and caption.** `save_figure(fig, stem, caption=…, notes=…, source=…,
   number=n)` writes PDF + PNG + `<stem>.caption.md`, plus a B&W proof to
   `<stem.parent>/gray/`. In a notebook, call `show_caption(...)` directly under
   the figure — same code as the sidecar, so the two cannot drift.
10. **Look at it.** Open the PNG, then open its `gray/` proof and confirm every
   series is still distinguishable. Check: nothing bold, no clipped text, no
   rotated tick labels where thinning ticks would do, text ≥ 7 pt at final size.

## Quick reference

| Need | Call |
|---|---|
| Style block / global | `econ_style()` · `use_econ_style()` |
| Size at print width | `figure_size("text" \| "journal" \| "column", ratio=…)` |
| **Bars** | **`bar(ax, …)` · `barh(ax, …)` — never `ax.bar`/`ax.barh`** |
| Year axis | `year_ticks(ax, first_year, last_year)` |
| Panel heading | `panel_title(ax, "A", "Title")` → "Panel A. Title" |
| Count / percent axes | `integer_ticks(ax, "y")` · `percent_ticks(ax, "y", xmax=100)` |
| Overall-mean line | `reference_line(ax, value, label=…)`; `in_legend=True` when bars crowd the label |
| Sparse bar labels | `value_labels(ax, bars, fmt="{:,.0f}")` |
| Series encodings | `GRAYS`, `HATCHES`, `LINESTYLES`, `MARKERS`, `OKABE_ITO` |
| Network marks | `network_axes(ax)` · `draw_node(...)` · `node_legend(...)` · `hairline(...)` · `arrow_props()` · `junction_dot(...)` |
| Node carrying an attribute | `box_node(ax, x, y, [name, attr], width=…, height=…)` — fills `BOX_FILL`; caller measures |
| Two-column flow | `flow_band(ax, x0, x1, y_left=…, y_right=…, thickness=…)` |
| Band separator | `separator_rule(ax, y)` |
| Save + caption | `save_figure(...)` · `show_caption(...)` · `caption_markdown(...)` |
| B&W check | the `gray/` proof `save_figure` writes · `grayscale_preview(png_path)` |

## Common mistakes

| Seen in unguided output | Journal convention |
|---|---|
| `ax.bar(...)` — solid black bars | `bar(ax, ...)`; the property cycle, not `patch.facecolor`, decides a bare bar's fill |
| A figure inventing its own gray (`GRAYS[2]` here, black there) | One `BAR_FILL` across the corpus; shades vary only *within* a stacked series |
| Bold label on the focus node of a network | The ink fill already distinguishes it; nothing is bold |
| Re-deriving `show_caption` / `year_ticks` per notebook | Import them; one definition or they drift |
| Bold, left-aligned "A. Title" or a big `suptitle` | Normal-weight "Panel A. Title", centred; figure title lives in the caption |
| Navy/teal/green bars, red reference line, coloured annotation text | Gray fills with black edges; black dashed reference line; all text in ink |
| Dashboard hero number ("66.3%" in 38 pt) | A proper panel: stacked counts or a share bar with an overall-share line |
| Value label on every bar of a 27-year series | No labels; the axis carries it. Label only short series |
| White text inside bars, "n of N" inside the bar | Small ink text above the bar, or move to the notes |
| Dashed gridlines, tinted background | No grid (or hairline solid gray); white background |
| Sans-serif at 10.5 pt on a 12 in canvas | Serif at 9 pt on a 6.5 in canvas |
| Source / methodology footer drawn inside the figure | `save_figure(..., notes=…, source=…)`; caption shown under the figure |
| Rotated year labels at every second year | Ticks every 4–5 years, horizontal |
