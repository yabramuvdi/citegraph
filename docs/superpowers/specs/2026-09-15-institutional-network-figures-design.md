# Annotated lineages and training→placement flows

Two additions to the paper 4 institutional figure set, plus the curated file and
the plotting primitives they need. Nothing existing is deleted or renumbered:
figures 1–7 and their alternates keep their stems, their captions and their
place in the manuscript.

## Why

The eleven existing figures split cleanly in two. Figures 1–4 and 7 give
**composition** — field, country, institution, decade, each as a bar. Figures 5
and 6 give **structure** — who trained whom, as a tidy tree and as a timeline.
The two never meet. A reader can learn that 31 of 56 doctorates were taken in
the United States, and separately that Bowles trained Cárdenas, but no figure
says that a lineage rooted at Amherst and Gothenburg now sits in Bogotá and
Medellín.

The reference figure that prompted this (Review of Political Economy, the
Victoria Chick network) closes exactly that gap by putting each person in a box
carrying name *and* institution. Figure 8 below is that move on our data.
Figures 9–10 close the other half: the joint distribution of where people
trained and where they now work, which figures 1–3 only show as separate
margins.

## What the data supports

Checked against `institutional/out/` on 2026-09-15.

| Fact | Consequence |
| ---- | ----------- |
| 62 PhD ties, 102 people, 40 components; 6 lineages of ≥5 hold 37 people | Figure 8 reuses figure 5's cut exactly |
| All 37 of those nodes resolve to an institution — no blanks | No "unknown" box in figure 8 |
| 20 of the 37 are in the author base; 17 are advisors outside it | Outside advisors need a different institution source |
| 44 of 56 advisors have no career record at all | Advisor affiliation is unavailable; the tie's institution is the only option |
| 2 advisors (Bowles, Johannesson) supervised at two institutions | Needs a stated tie-break rule |
| Institution names run to 50 characters | Boxes need curated short names |
| 56 authors have both a PhD country and an affiliation country, and both fields | Figures 9 and 10 share one denominator |
| `career_positions.csv` has **zero** year values on `affiliation` rows | Career-trajectory timelines are not buildable — see Ruled out |

## Figure 8 — annotated doctoral lineages

`fig8_lineages_annotated`, written alongside `fig5_doctoral_lineages`, which
stays as the clean structural read.

**Same population as figure 5:** the 6 lineages with ≥5 people, 37 nodes over 3
generations. Holding the cut fixed is the point — the two figures are meant to
be compared, and the ≥3 cut (64 people) does not fit a page once nodes become
two-line boxes.

**Node content.** Each node is a rectangle with the person's name on the first
line and an institution on the second, smaller and in `GRAYS[1]`:

- **In the author base (20):** current affiliation. All 20 have one.
- **Outside advisor (17):** the institution recorded on their supervision tie —
  that is, where the doctorate they supervised was awarded. Present on all 62
  ties, so no node is empty.
- **Two ties to break.** Bowles supervised at Amherst (2000) and Stockholm
  University (2009); Johannesson at Stockholm University (twice) and the
  Stockholm School of Economics. Rule: **most frequent institution, earliest
  year breaks a tie** → Bowles shows Amherst, Johannesson shows Stockholm
  University. The caption states the rule and names the two people, because a
  box that silently drops a second institution is a claim the data does not
  make.

**Encoding.** Box facecolor carries membership and nothing else — `NODE_FILL`
for someone in the base, `NODE_OPEN` for an advisor outside it — matching
`draw_node` and the `styling-econ-figures` skill. The institution is text, never
a second colour channel. No bold, no per-lineage colour.

**Layout.** The existing dendrogram router is reused rather than forked. It is
already driven by exactly two quantities per node: a right extent, currently
`LABEL_OFF + widths[n]`, and a left arrow anchor, currently
`x[layer] - NODE_R - 0.6`. `_columns`, `_fit_fontsize` and `_route` in the
notebook are generalised to take an `extent(n) -> (left, right)` callback, so
the dot tree and the box tree share one crossing-free router and cannot drift
apart. Row height becomes box height plus gap instead of `fs * 1.80`. Column
width becomes the widest box in that generation instead of the longest label.

**Expected size.** 37 boxes, 3 generations, two text lines each — roughly 1.6×
the height of figure 5, so a full-page figure at text width. The existing
`_fit_fontsize` binary search keeps it inside the printed width.

## Figures 9 and 10 — training → placement

**Figure 9, `fig9_training_placement_country`.** Two stacked columns of blocks,
PhD country on the left and current affiliation country on the right, joined by
straight-edged ribbons of width proportional to the count. N = 56.

Category rule: a country is named when it reaches **3 or more on either side**
— United States, United Kingdom, Sweden, Colombia, Italy, Spain — and everything
else pools into "Other". Applying one threshold across both sides keeps a
country from appearing on one side and vanishing on the other.

The finding is already legible in the data: 35 of the 56 work in the country
that awarded their doctorate, 22 of them United States → United States. The
asymmetry is Colombia's — it awarded **3** of the 56 doctorates and employs
**14** of the 56, and **6 United States doctorates now work there**.

**Figure 10, `fig10_training_placement_field`.** The same construction over
fields: 6 PhD fields to 8 affiliation fields, N = 56, no pooling needed. 16 of
56 changed field between doctorate and current post — Engineering into
Economics, Economics into Political Science and into Environment.

**Ribbon encoding.** Stayers (same category both sides) take the lightest fill,
`GRAYS[3]`; movers take `BAR_FILL`; both keep a hairline black edge. A legend
names the two. This is what keeps the figure alive in the grayscale proof — an
undifferentiated alluvial turns to mush there, and the movers are the whole
story. Ribbons are painted **largest first**, so no small flow is buried, and
the paint order is derived from a sorted list rather than dict order.

**Figure 9alt, `fig9alt_training_placement_matrix`.** The same country data as a
matrix of proportional squares, PhD country down, affiliation country across,
with marginal bars. Written to an `alt` stem and *not* to the canonical one,
following the convention figures 5, 6 and 7 already use: the manuscript keeps
pointing at figure 9 until someone chooses to swap it. It exists because the
flow's one real weakness is ribbon overlap at seven categories a side, and the matrix
has no overlap by construction. Worth seeing side by side before the figure goes
into a manuscript.

## Institution short names — a new curated file

Boxes cannot carry "London School of Economics and Political Science". They need
"LSE", the way the reference figure says "Unicamp" and "UFMG".

**New file: `institutional/institution_short_names.csv`**, columns
`ror_id,canonical_name,short_name,note`.

- **Keyed on ROR id, not on the raw string.** `crosswalk_institutions.csv` has
  106 rows for ~85 institutions — several raw spellings map to one institution —
  so a short name attached to a raw string could give one institution two
  different labels depending on which spelling a row happened to use. The single
  institution with no ROR record (noted already in figure 2's caption) keys on
  `canonical_name` instead.
- **Separate from the crosswalk on purpose.** The crosswalk answers *which
  institution is this string*; this file answers *how do we print it*. That is
  the same separation `journal_aliases.csv` and `author_external_ids.csv` keep
  in the main package, and the reason CLAUDE.md gives for it — two curated files
  must not both be in charge of one decision.
- **23 rows are needed for figure 8**; I will fill a first pass for the full
  ~85 and mark every one `note=generated` so you and Maria can see at a glance
  which have been checked. Corrected rows get the note cleared.
- **Failure behaviour, mirroring `unused_aliases` in `journals.py`:** a row whose
  `ror_id` matches no institution in the base raises, because a curated row that
  silently does not apply is worse than no row. An institution that is *drawn*
  and has no short name falls back to its canonical name and is reported, so the
  gap is visible rather than silently ugly.
- `build_clean_base.py` joins the column into `career_positions.csv`,
  `base_institucional_clean.csv` (as `*_institution_short`) and
  `supervision.csv` (as `institution_short`). Figure 7 picks up shorter axis
  labels for free.

`make_crosswalks.py` appends only unseen `raw_key`s to the existing crosswalk,
so it will not disturb this file; the new file is not generated by it at all.

## Code layout

**`src/citegraph/plotting.py`** gains the marks, not the layout — the same
boundary the module already keeps, where it owns the vocabulary and the notebook
owns measurement:

- `box_node(ax, x, y, lines, *, width, height, inside=True, focus=False, ...)`
  — a rectangle at a caller-measured size with one or two text lines, fill
  following `draw_node`'s membership rule, black edge at 0.7. Width and height
  arrive in data units; the notebook keeps the measurement problem, as the
  comment in its layout cell already explains.
- `node_legend(..., shape="box")` — the existing legend with square proxies, so
  figure 8's legend matches its marks.
- `flow_band(ax, x0, x1, y_left, y_right, thickness, *, fill=BAR_FILL, ...)` —
  one straight-edged ribbon as a filled quadrilateral with a hairline edge.
  `y_left` and `y_right` are the ribbon's lower edge where it leaves and where
  it arrives; `thickness` is the same at both ends, because both ends are the
  same count.

**`examples/paper4_institutional_figures.ipynb`** gains three sections — "Figure
8. Lineages with institutions", "Figures 9–10. Training and placement", and the
matrix alternative under figure 9 — plus the router generalisation described
above. Assembly stays in the notebook, matching how figures 5 and 6 are built.

**`institutional/build_clean_base.py`** joins the short-name column and extends
its `PASS` check to cover it.

**Correction folded in:** 14 caption strings in the notebook say "59 authors" or
"59-author list". The base has held 58 since Cleve E. Willis was removed. Every
one becomes 58. These are captions of figures being regenerated anyway, and a
wrong N in a manuscript caption is a real defect.

## Testing and verification

- **Unit tests** for `box_node`, `flow_band` and the box legend in
  `tests/test_plotting.py`, alongside the existing `draw_node` tests: fill
  follows membership, the box honours the width and height it is given, a ribbon
  closes, the legend proxies match the marks.
- **A short-name fixture test** covering the raise on an unmatched `ror_id` and
  the fallback-plus-report on a missing one.
- **Figure verification** is the house procedure: execute the notebook twice in
  separate processes with the pyenv 3.11.10 interpreter and `cmp` the PNGs byte
  for byte. Any new node or edge iteration is built with `sorted()` at
  construction, because `subgraph` and `set` iteration follow string hashing and
  a fixed `seed=` does not save a layout from it.
- **Grayscale proof** for all four new figures lands under `figures/gray/`
  automatically through `save_figure`; the ribbon fills are chosen for that
  check specifically.

## Ruled out, with the evidence

- **Career-trajectory timelines (Gantt per person).** `career_positions.csv`
  carries no year at all on `affiliation` rows and years on only 23 of 58
  postdoc rows. There is nothing to place on a time axis but a single PhD point.
- **Institution-level lineage trees.** Would need each advisor's own doctorate;
  44 of 56 advisors have no career record.
- **Node size by citations on figure 8.** Only 6 of the 44 outside advisors have
  a citegraph author id, so the entire advisor generation would render at
  minimum size and read as "unimportant" when it is really "unmeasured". The
  same encoding is safe on figure 6, where all 22 people are in the base — left
  for a later pass.
- **A geographic map.** ROR gives country, not coordinates, without a network
  fetch this pipeline does not otherwise need.
