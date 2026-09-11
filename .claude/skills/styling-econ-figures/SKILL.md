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
6. **Marks.** Bars: gray fill with black edge (the rc does this), width 0.7–0.8,
   baseline at zero, `integer_ticks` / `percent_ticks`. Two or more series:
   `GRAYS` + `HATCHES` and a frameless legend. `value_labels` only where the
   exact number matters (a short ranking, five-year totals); never on a dense
   annual series, never bold, never inside a bar.
7. **Benchmarks** as `reference_line(ax, value, label="All periods: 66%")` —
   dashed black hairline with a small ink label at its right end. If any bar or
   value label sits near the line at the right edge, pass `in_legend=True` and
   call `ax.legend()` instead of nudging text or adding white halos.
8. **Tick labels stay horizontal.** Dense years: a tick every 4–5 years.
   Five-year periods: two-line labels (`"1995–\n1999"`) or abbreviated
   (`"1995–99"`). Rotation is the last resort, not the first.
9. **Save and caption.** `save_figure(fig, stem, caption=…, notes=…, source=…,
   number=n)` writes PDF + PNG + `<stem>.caption.md`. In a notebook, show
   `Markdown(caption_markdown(...))` directly under the figure.
10. **Look at it.** Open the PNG. Run `grayscale_preview(png)` and confirm every
   series is still distinguishable. Check: nothing bold, no clipped text, no
   rotated tick labels where thinning ticks would do, text ≥ 7 pt at final size.

## Quick reference

| Need | Call |
|---|---|
| Style block / global | `econ_style()` · `use_econ_style()` |
| Size at print width | `figure_size("text" \| "journal" \| "column", ratio=…)` |
| Panel heading | `panel_title(ax, "A", "Title")` → "Panel A. Title" |
| Count / percent axes | `integer_ticks(ax, "y")` · `percent_ticks(ax, "y", xmax=100)` |
| Overall-mean line | `reference_line(ax, value, label=…)`; `in_legend=True` when bars crowd the label |
| Sparse bar labels | `value_labels(ax, bars, fmt="{:,.0f}")` |
| Series encodings | `GRAYS`, `HATCHES`, `LINESTYLES`, `MARKERS`, `OKABE_ITO` |
| Save + caption | `save_figure(...)` · `caption_markdown(...)` |
| B&W check | `grayscale_preview(png_path)` |

## Common mistakes

| Seen in unguided output | Journal convention |
|---|---|
| Bold, left-aligned "A. Title" or a big `suptitle` | Normal-weight "Panel A. Title", centred; figure title lives in the caption |
| Navy/teal/green bars, red reference line, coloured annotation text | Gray fills with black edges; black dashed reference line; all text in ink |
| Dashboard hero number ("66.3%" in 38 pt) | A proper panel: stacked counts or a share bar with an overall-share line |
| Value label on every bar of a 27-year series | No labels; the axis carries it. Label only short series |
| White text inside bars, "n of N" inside the bar | Small ink text above the bar, or move to the notes |
| Dashed gridlines, tinted background | No grid (or hairline solid gray); white background |
| Sans-serif at 10.5 pt on a 12 in canvas | Serif at 9 pt on a 6.5 in canvas |
| Source / methodology footer drawn inside the figure | `save_figure(..., notes=…, source=…)`; caption shown under the figure |
| Rotated year labels at every second year | Ticks every 4–5 years, horizontal |
