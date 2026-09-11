# Figure conventions in top economics journals

What AER, QJE, JPE, Econometrica and REStud actually print, with the source
for each rule where one exists. Rules without a quoted source are observed
practice across published articles and the Stata `plotplain`/`s1mono` schemes
economists overwhelmingly use.

## Sources

- **AER Style Guide** (aeaweb.org/journals/aer/style-guide): figures "must be
  supplied as vector-based graphics … vector PDF, EPS, AI, WMF, or PPT";
  "use letters for indicating individual panels"; "Source notes, if any,
  should be placed after other notes related to the figure"; sections of a
  table are denoted "Panel A, Panel B, etc." (figures follow the same
  wording in print); variables in figures are italic, vectors/matrices bold;
  raster images at 300 dpi; decimals written `0.357`, never `.357`.
- **Schwabish, J. (2014), "An Economist's Guide to Visualizing Data",
  *Journal of Economic Perspectives* 28(1): 209–234.** Principles: *show the
  data*, *reduce the clutter*, *integrate the text and the graph*, *avoid the
  spaghetti chart*, *start with gray*. Bars and columns start at zero; no 3D;
  avoid dual axes and pies; direct labels over legends where they fit.
- **Bischof, D. (2017), "New graphic schemes for Stata: plotplain and
  plottig", *Stata Journal* 17(3): 748–759.** `plotplain`: no background
  tinting, thin axes, small fonts, small markers, medium-thin lines, unframed
  legends, "shades of gray and shapes of lines to distinguish plotted
  subgroups"; `plotplainblind` adds the Okabe–Ito colour-blind palette.
- **Okabe, M. & Ito, K. (2008), "Color Universal Design"**: the eight-colour
  CVD-safe palette (`OKABE_ITO` in `citegraph.plotting`). Validated with the
  `dataviz` skill's palette checker: adjacent-pair CVD ΔE ≥ 7.6, normal-vision
  ΔE ≥ 21; orange, sky blue and reddish purple sit below 3:1 contrast on white
  and therefore always travel with a dash, marker or hatch.

## Layout and captions

- **No title inside the figure.** The caption reads "Figure 1. Title", set by
  the journal. Everything the reader needs to interpret the plot goes in
  *Notes:* after the caption; *Source:* comes last (AER). In a notebook,
  render the same block as Markdown directly under the figure.
- **Panels** are headed "Panel A. Title" in normal weight, centred above each
  axes (AER). Econometrica prints "(a)" below; either is acceptable, but be
  consistent within a paper. `panel_title` gives the AER form.
- **Legend**: frameless, inside the axes where there is empty space, otherwise
  below the plot in one row. A single series gets no legend.
- **No in-figure source footers, subtitles, or "key insight" annotations.**
  A single benchmark line with a short label ("All periods: 66%") is the most
  annotation a journal figure carries.

## Type

- One family throughout, matching the manuscript body: serif (Times / STIX)
  for LaTeX or Word manuscripts; Helvetica/Arial is the Stata default and also
  common. Never mix.
- 8–10 pt at final printed size; tick labels one step smaller than axis
  labels. Nothing bold, nothing italic except variable names.
- Sentence case for axis labels and panel titles ("Number of publications",
  not "Number Of Publications"); units in parentheses or the label itself
  ("Share of publications (%)").
- Numbers: thousands separators, leading zero on decimals, integers on count
  axes, percent sign on share axes.

## Colour

- **Design for grayscale first.** Print editions and photocopies are
  black-and-white; many referees print. Every series must be identifiable
  with colour removed: dash pattern for lines, hatch or gray step for bars,
  marker shape for scatter.
- Single-series bars: mid-dark gray (`GRAYS[1]`) with a black edge, or solid
  black for a small number of bars.
- Two to four series: `GRAYS` steps dark → light, optionally with `HATCHES`.
- Colour (`palette="color"`) only when more than three series must be told
  apart at a glance; use `OKABE_ITO` in fixed order, never a rainbow, never
  Matplotlib's default tab10, never a colour whose only job is decoration.
- Text is always ink (`INK`). Never colour a label to match its series.

## Marks

- Bars: width 0.7–0.8 of the slot, baseline at zero, thin black edge. No
  rounded corners, no gradients, no alpha.
- Lines: 1–1.5 pt, distinct dashes, small markers (≤ 4 pt) only where points
  are few. Confidence bands: light gray fill or dashed lines, never a colour
  ribbon.
- Value labels: sparse and small. Horizontal rankings of ≤ 20 items may carry
  a count at the bar end; dense time series never do.
- Error bars with 2 pt caps; whiskers in ink.

## Axes and chrome

- Left and bottom spines only, 0.6 pt, black. Ticks outward, 3 pt.
- No gridlines by default. If a panel needs them (many bars, wide range), a
  single hairline solid gray horizontal grid behind the data.
- No background tinting; pure white surface.
- Tick labels horizontal. Thin ticks before rotating labels: a 27-year series
  gets a tick every four or five years, not a rotated label on each. Period
  labels that would collide are broken over two lines ("1995–" / "1999") or
  abbreviated ("1995–99"), never rotated 45°.
- A benchmark line's label may collide with bars that rise above the line at
  the right edge; in that case the line takes a legend entry ("All periods:
  66%") rather than free text with a white halo.
- Bar charts and count axes start at zero. Line charts may not, but say so in
  the notes.

## Sizes

Design at the size the figure will be printed, so the type is right without
rescaling:

| Preset | Width | Use |
|---|---|---|
| `text` | 6.5 in | full text width of a US-letter working paper (1 in margins) |
| `journal` | 4.75 in | approximate AER / QJE text block |
| `column` | 3.25 in | half-width panel or two-column layout |

Heights: single panel ≈ 0.6 × width; two panels side by side ≈ 0.42–0.5 ×
width; a 20-row horizontal bar chart ≈ 0.8 × width.

## Output

- Vector PDF is the deliverable (AER: "vector PDF, EPS, AI, WMF, or PPT").
  `pdf.fonttype = 42` embeds TrueType fonts; Type 3 fonts (the Matplotlib
  default) are rejected by several publishers.
- PNG at 300 dpi for Word drafts and slides.
- A `<stem>.caption.md` beside the artwork keeps title, notes and source with
  the file so the manuscript and the notebook show the same text.
- File names carry the panel letter when panels are shipped as separate files
  (AER: `Figure1a.pdf`, `Figure1b.pdf`).

## How this differs from the `dataviz` skill

`dataviz` targets interactive screen dashboards. Its form heuristic, one-axis
rule, fixed-order categorical palette, and anti-pattern list all still apply
here. The differences for print: black and gray are the *primary* series
colours (its lightness-band and chroma-floor checks assume chromatic series);
hover/tooltip and dark-mode steps do not apply; text is serif to match the
manuscript rather than a UI sans; rounded bar ends and 2 px white gaps are
replaced by square bars with a thin black edge; hero numbers and stat tiles
are never used.
