"""Publication-quality matplotlib styling for economics manuscripts.

The conventions encoded here follow what top economics journals (AER, QJE,
JPE, Econometrica, REStud) actually print:

* **Grayscale-first.** Black and gray marks carry the data; colour, when
  used, is the colour-blind-safe Okabe–Ito set and is always doubled by a
  line style so the figure survives black-and-white printing.
* **No title inside the figure.** The title lives in the caption
  ("Figure 1. …"), followed by *Notes:* then *Source:*. Panels are labelled
  "Panel A. …" above each axes.
* **Serif type matching the manuscript**, 8–10 pt at final size, one
  weight. Figures are designed at the final printed width so nothing is
  rescaled.
* **Minimal chrome.** Left and bottom spines only, thin black axes,
  outward ticks, no gridlines, bars start at zero, integer/percent tick
  formats.
* **Vector output** (PDF with TrueType fonts embedded) plus a 300 dpi PNG.

``matplotlib`` is an optional dependency (``pip install "citegraph[plots]"``);
it is imported lazily inside each function so ``import citegraph`` stays
cheap.

Typical use inside a notebook::

    from citegraph.plotting import bar, econ_style, figure_size, panel_title, save_figure

    with econ_style():
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figure_size("text", ratio=0.45))
        bar(ax_a, years, counts)
        panel_title(ax_a, "A", "Annual counts")
        ...
        save_figure(fig, out_dir / "figures" / "trend",
                    caption="Publication trends", notes="…", source="…")

Draw bars with :func:`bar` / :func:`barh` rather than ``ax.bar`` / ``ax.barh``:
matplotlib takes an unspecified bar colour from the property cycle, whose first
entry is ink, so a bare ``ax.bar`` comes out solid black under this style
however ``patch.facecolor`` is set. Network figures have their own vocabulary —
:func:`network_axes`, :func:`draw_node`, :func:`hairline` — so a node-link
diagram and a bar chart in the same manuscript read as one family.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.text import Annotation

__all__ = [
    "INK",
    "GRAYS",
    "OKABE_ITO",
    "LINESTYLES",
    "MARKERS",
    "HATCHES",
    "WIDTHS",
    "BAR_FILL",
    "BAR_WIDTH",
    "NODE_FILL",
    "NODE_OPEN",
    "NODE_EDGE",
    "NODE_RADIUS",
    "LABEL_OFFSET",
    "EDGE_COLOR",
    "RULE_COLOR",
    "econ_rc",
    "econ_style",
    "use_econ_style",
    "figure_size",
    "bar",
    "barh",
    "year_ticks",
    "panel_title",
    "integer_ticks",
    "percent_ticks",
    "reference_line",
    "value_labels",
    "caption_markdown",
    "save_figure",
    "show_caption",
    "grayscale_preview",
    "draw_node",
    "box_node",
    "flow_band",
    "node_legend",
    "hairline",
    "arrow_props",
    "network_axes",
    "separator_rule",
]

# ----------------------------------------------------------------------
# Palette constants
# ----------------------------------------------------------------------

#: Text, axes, and the first series.
INK = "#000000"

#: Ordinal gray ramp, dark → light, for bars/stacked segments. Fills always
#: carry a black edge so the lightest step still reads on white.
GRAYS: tuple[str, ...] = ("#252525", "#636363", "#969696", "#BDBDBD", "#E6E6E6")

#: Okabe–Ito colour-blind-safe hues in fixed order (blue, vermillion,
#: bluish green, reddish purple, orange, sky blue). Validated: adjacent-pair
#: CVD ΔE ≥ 7.6, normal-vision ΔE ≥ 21. The last three are light against white
#: and must always be paired with a line style, marker, or direct label.
OKABE_ITO: tuple[str, ...] = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")

#: Dash patterns in the order they are assigned to series.
LINESTYLES: tuple[str, ...] = ("-", "--", ":", "-.")

#: Marker shapes in the order they are assigned to series.
MARKERS: tuple[str, ...] = ("o", "s", "^", "D")

#: Hatch patterns for distinguishing bar series without colour. The empty
#: string is the first (solid) series.
HATCHES: tuple[str, ...] = ("", "////", "....", "xxxx", "\\\\\\\\")

#: Default bar fill. ``ax.bar`` takes its colour from the property cycle, whose
#: first entry is :data:`INK`, so an un-styled bar comes out solid black however
#: ``patch.facecolor`` is set — no rcParam can express "black lines, gray bars".
#: :func:`bar` and :func:`barh` apply this fill instead; call them rather than
#: ``ax.bar``/``ax.barh`` so every bar in the corpus reads the same.
BAR_FILL = GRAYS[1]

#: Default bar thickness (fraction of the category step).
BAR_WIDTH = 0.72

#: Network node fill for a node inside the population being described.
NODE_FILL = GRAYS[1]

#: Network node fill for a node outside it (an advisor who is not a listed
#: author, a co-cited author below the inclusion threshold).
NODE_OPEN = "white"

#: Every network node carries this edge so the open fill still reads on white.
NODE_EDGE = INK

#: Network edges and connectors: a hairline dark enough to follow across the
#: page but light enough not to compete with the node markers.
EDGE_COLOR = "#555555"

#: Hairline separating bands of a grouped figure (lineage blocks, tie rows).
RULE_COLOR = "#DBDBDB"

#: Network node radius in points. Markers are sized ``2 * NODE_RADIUS``.
NODE_RADIUS = 2.5

#: Gap in points between a node centre and its label.
LABEL_OFFSET = 6.0

#: Final printed widths in inches. ``text`` is a US-letter manuscript with
#: 1-inch margins; ``journal`` approximates an AER/QJE text block; ``column``
#: is one column of a two-column layout or a half-width panel.
WIDTHS: dict[str, float] = {"column": 3.25, "journal": 4.75, "text": 6.5}

_SERIF_STACK = (
    "Times New Roman",
    "Times",
    "STIX Two Text",
    "STIXGeneral",
    "Nimbus Roman",
    "DejaVu Serif",
)
_SANS_STACK = ("Helvetica", "Arial", "Liberation Sans", "DejaVu Sans")


# ----------------------------------------------------------------------
# Lazy import helpers
# ----------------------------------------------------------------------


def _mpl():
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "citegraph.plotting requires matplotlib. Install with: "
            'pip install "citegraph[plots]"'
        ) from exc
    return matplotlib


def _pyplot():
    _mpl()
    import matplotlib.pyplot as plt

    return plt


# ----------------------------------------------------------------------
# rcParams
# ----------------------------------------------------------------------


def _prop_cycle(palette: str):
    from cycler import cycler

    if palette == "mono":
        colors = (INK, "#636363", "#A6A6A6", "#404040")
        styles = LINESTYLES
    elif palette == "color":
        colors = (INK, *OKABE_ITO)
        # Repeat the dash cycle so every colour still differs in line style.
        styles = tuple(LINESTYLES[i % len(LINESTYLES)] for i in range(len(colors)))
    else:
        raise ValueError(f"Unknown palette {palette!r}; expected 'mono' or 'color'.")
    return cycler(color=list(colors)) + cycler(linestyle=list(styles))


def econ_rc(
    *,
    palette: str = "mono",
    font: str = "serif",
    base_size: float = 9.0,
) -> dict[str, Any]:
    """Return the rcParams dict for the economics-journal style.

    Parameters
    ----------
    palette:
        ``"mono"`` (default) cycles black and grays with distinct dashes;
        ``"color"`` cycles black then the Okabe–Ito hues, still with dashes.
    font:
        ``"serif"`` (Times/STIX, matching a LaTeX or Word manuscript) or
        ``"sans"`` (Helvetica/Arial, the Stata default many journals print).
    base_size:
        Body font size in points at the final printed width. Tick labels,
        legend entries and value labels are one step smaller.
    """
    _mpl()
    if font == "serif":
        family, stack, mathtext = "serif", _SERIF_STACK, "stix"
    elif font == "sans":
        family, stack, mathtext = "sans-serif", _SANS_STACK, "stixsans"
    else:
        raise ValueError(f"Unknown font {font!r}; expected 'serif' or 'sans'.")

    small = round(base_size - 1.0, 2)
    return {
        # --- type ---
        "font.family": family,
        f"font.{family}": list(stack),
        "mathtext.fontset": mathtext,
        "font.size": base_size,
        "font.weight": "normal",
        "axes.titlesize": base_size,
        "axes.titleweight": "normal",
        "axes.titlelocation": "center",
        "axes.titlepad": 6.0,
        "axes.labelsize": base_size,
        "axes.labelweight": "normal",
        "axes.labelpad": 4.0,
        "xtick.labelsize": small,
        "ytick.labelsize": small,
        "legend.fontsize": small,
        "legend.title_fontsize": small,
        "figure.titlesize": base_size,
        "figure.titleweight": "normal",
        "axes.unicode_minus": True,
        # --- axes & ticks ---
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.axisbelow": True,
        "axes.grid": False,
        "grid.color": "#BFBFBF",
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "axes.prop_cycle": _prop_cycle(palette),
        # --- marks ---
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "lines.markeredgewidth": 0.6,
        "patch.linewidth": 0.6,
        "patch.edgecolor": INK,
        "patch.force_edgecolor": True,
        "hatch.linewidth": 0.5,
        "hatch.color": INK,
        "errorbar.capsize": 2.0,
        # --- legend ---
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "legend.handleheight": 0.7,
        "legend.borderaxespad": 0.3,
        "legend.labelspacing": 0.4,
        "legend.columnspacing": 1.2,
        # --- figure & output ---
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "figure.dpi": 130,
        "figure.constrained_layout.use": True,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


@contextlib.contextmanager
def econ_style(
    *,
    palette: str = "mono",
    font: str = "serif",
    base_size: float = 9.0,
    overrides: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Context manager applying :func:`econ_rc` for the duration of a block.

    ``overrides`` are merged last, so callers can switch on gridlines or
    change a single size without redefining the whole style.
    """
    plt = _pyplot()
    rc = econ_rc(palette=palette, font=font, base_size=base_size)
    if overrides:
        rc.update(overrides)
    with plt.rc_context(rc):
        yield rc


def use_econ_style(
    *,
    palette: str = "mono",
    font: str = "serif",
    base_size: float = 9.0,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply :func:`econ_rc` globally (for the rest of a notebook session)."""
    plt = _pyplot()
    rc = econ_rc(palette=palette, font=font, base_size=base_size)
    if overrides:
        rc.update(overrides)
    plt.rcParams.update(rc)
    return rc


# ----------------------------------------------------------------------
# Sizes
# ----------------------------------------------------------------------


def figure_size(
    width: str | float = "text",
    *,
    ratio: float = 0.6,
    height: float | None = None,
) -> tuple[float, float]:
    """Return ``(width, height)`` in inches at the final printed size.

    ``width`` is a preset name from :data:`WIDTHS` or a number of inches.
    ``height`` overrides ``ratio`` (height / width) when given.
    """
    if isinstance(width, str):
        try:
            w = WIDTHS[width]
        except KeyError as exc:
            raise ValueError(
                f"Unknown width preset {width!r}; expected one of {sorted(WIDTHS)} or inches."
            ) from exc
    else:
        w = float(width)
    h = float(height) if height is not None else w * ratio
    return (w, h)


# ----------------------------------------------------------------------
# Axis helpers
# ----------------------------------------------------------------------


def _axis(ax: Axes, axis: str):
    if axis == "x":
        return ax.xaxis
    if axis == "y":
        return ax.yaxis
    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def bar(ax: Axes, x, height, *, width: float = BAR_WIDTH, **kwargs: Any) -> BarContainer:
    """Vertical bars in the house fill: :data:`BAR_FILL` with a black edge.

    Use instead of ``ax.bar``. Matplotlib takes an un-specified bar colour from
    the property cycle, whose first entry is black, so ``ax.bar`` produces solid
    black bars under :func:`econ_style` no matter what ``patch.facecolor`` says.
    Passing ``color=`` here still wins, for stacked or multi-series bars.
    """
    kwargs.setdefault("color", BAR_FILL)
    kwargs.setdefault("edgecolor", INK)
    return ax.bar(x, height, width=width, **kwargs)


def barh(ax: Axes, y, width, *, height: float = BAR_WIDTH, **kwargs: Any) -> BarContainer:
    """Horizontal bars in the house fill. See :func:`bar`."""
    kwargs.setdefault("color", BAR_FILL)
    kwargs.setdefault("edgecolor", INK)
    return ax.barh(y, width, height=height, **kwargs)


def year_ticks(ax: Axes, first_year: int, last_year: int, *, pad: float = 0.7) -> None:
    """Horizontal year ticks at a readable spacing, with the x-limits padded.

    Every five years over a long span, every two over a short one, so a dense
    annual series never needs rotated labels. Ticks land on round years.
    """
    step = 5 if last_year - first_year > 15 else 2
    first_tick = first_year + (-first_year) % step
    ax.set_xticks(range(first_tick, last_year + 1, step))
    ax.set_xlim(first_year - pad, last_year + pad)


def panel_title(ax: Axes, letter: str, title: str | None = None, **kwargs: Any):
    """Label a panel the AER way: ``Panel A. Title`` centred above the axes."""
    text = f"Panel {letter}. {title}" if title else f"Panel {letter}"
    return ax.set_title(text, **kwargs)


def integer_ticks(ax: Axes, axis: str = "y") -> None:
    """Force whole-number ticks (counts) on one axis."""
    from matplotlib.ticker import MaxNLocator

    _axis(ax, axis).set_major_locator(MaxNLocator(integer=True))


def percent_ticks(ax: Axes, axis: str = "y", *, xmax: float = 100, decimals: int = 0) -> None:
    """Format ticks as percentages. ``xmax=100`` for data already in percent,
    ``xmax=1`` for proportions."""
    from matplotlib.ticker import PercentFormatter

    _axis(ax, axis).set_major_formatter(PercentFormatter(xmax=xmax, decimals=decimals))


def reference_line(
    ax: Axes,
    value: float,
    *,
    label: str | None = None,
    orientation: str = "h",
    color: str = INK,
    linestyle: str = "--",
    linewidth: float = 0.8,
    in_legend: bool = False,
    label_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Line2D:
    """Draw a dashed hairline at ``value`` (a mean, a target, an overall share).

    By default the optional ``label`` sits just above the line at its right
    end (or beside a vertical line at its top) in small ink text — the journal
    way to annotate a benchmark without a legend entry. When bars or lines
    would collide with that text, pass ``in_legend=True``: the label is then
    attached to the line artist so a later ``ax.legend()`` lists it instead.
    """
    if orientation == "h":
        line = ax.axhline(value, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)
    elif orientation == "v":
        line = ax.axvline(value, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)
    else:
        raise ValueError(f"orientation must be 'h' or 'v', got {orientation!r}")

    if label and in_legend:
        line.set_label(label)
    elif label:
        opts: dict[str, Any] = {"color": INK, "fontsize": "small"}
        if orientation == "h":
            opts.update(
                x=0.995,
                y=value,
                ha="right",
                va="bottom",
                transform=ax.get_yaxis_transform(),
            )
        else:
            opts.update(
                x=value,
                y=0.98,
                ha="left",
                va="top",
                rotation=90,
                transform=ax.get_xaxis_transform(),
            )
        if label_kwargs:
            opts.update(label_kwargs)
        ax.text(s=label, **opts)
    return line


def value_labels(
    ax: Axes,
    bars: BarContainer,
    *,
    fmt: str = "{:,.0f}",
    labels: Iterable[str] | None = None,
    padding: float = 2.0,
    **kwargs: Any,
) -> list[Annotation]:
    """Annotate bars with their values in small, regular-weight ink text.

    Use sparingly (a handful of bars, or a horizontal ranking where the exact
    count matters). Never bold, never coloured.
    """
    if labels is None:
        labels = [fmt.format(value) for value in bars.datavalues]
    opts: dict[str, Any] = {
        "color": INK,
        "fontsize": "small",
        "fontweight": "normal",
        "padding": padding,
    }
    opts.update(kwargs)
    return ax.bar_label(bars, labels=list(labels), **opts)


# ----------------------------------------------------------------------
# Captions and saving
# ----------------------------------------------------------------------


def caption_markdown(
    caption: str,
    *,
    notes: str | None = None,
    source: str | None = None,
    number: int | str | None = None,
) -> str:
    """Render a journal-style caption block as Markdown.

    ``**Figure N. Caption**`` then ``*Notes:*`` then ``*Source:*`` — notes
    precede the source note, following the AER style guide. Without ``number``
    the title stands alone (``**Caption**``) so figures can be numbered later,
    when they are arranged in the manuscript.
    """
    title = caption.strip()
    heading = f"Figure {number}. {title}" if number is not None else title
    parts = [f"**{heading}**"]
    if notes:
        parts.append(f"*Notes:* {notes.strip()}")
    if source:
        parts.append(f"*Source:* {source.strip()}")
    return "\n\n".join(parts)


def save_figure(
    fig: Figure,
    stem: str | Path,
    *,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 300,
    caption: str | None = None,
    notes: str | None = None,
    source: str | None = None,
    number: int | str | None = None,
    grayscale: bool = True,
) -> list[Path]:
    """Save ``fig`` as ``<stem>.<fmt>`` for each format, plus ``<stem>.caption.md``.

    PDF is the vector file journals ask for (TrueType fonts embedded via
    ``pdf.fonttype = 42``); PNG at ``dpi`` is for Word drafts and slides.
    The caption file keeps title, notes and source next to the artwork so the
    figure itself can stay free of in-plot titles.

    ``grayscale`` (default) also writes a black-and-white proof to a ``gray/``
    subdirectory, so the "legible without colour" check leaves evidence on disk
    without doubling the file count in the directory handed to a coauthor. It is
    skipped silently when Pillow is absent or no PNG was requested.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        path = stem.with_name(f"{stem.name}.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        written.append(path)
    if caption:
        caption_path = stem.with_name(f"{stem.name}.caption.md")
        caption_path.write_text(
            caption_markdown(caption, notes=notes, source=source, number=number) + "\n",
            encoding="utf-8",
        )
        written.append(caption_path)
    if grayscale:
        png = next((p for p in written if p.suffix == ".png"), None)
        if png is not None:
            gray_dir = stem.parent / "gray"
            try:
                gray_dir.mkdir(parents=True, exist_ok=True)
                written.append(
                    grayscale_preview(png, gray_dir / f"{stem.name}.gray.png")
                )
            except ImportError:  # Pillow not installed; the proof is optional
                pass
    return written


def network_axes(ax: Axes, *, equal: bool = True) -> Axes:
    """Strip a network drawing down to its marks: no spines, ticks, or frame.

    A node-link diagram has no meaningful axes, so the chrome
    :func:`econ_style` draws for a plot is noise here. ``equal`` keeps circles
    circular; pass ``equal=False`` when one axis carries real units (a year).
    """
    ax.set_axis_off()
    if equal:
        ax.set_aspect("equal")
    return ax


def hairline(
    ax: Axes,
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    color: str = EDGE_COLOR,
    linewidth: float = 0.7,
    zorder: float = 2.0,
    **kwargs: Any,
) -> Line2D:
    """Draw one thin connector segment.

    ``econ_style`` installs a property cycle that carries *line styles* as well
    as colours, so a bare ``ax.plot`` silently comes out dashed or dotted on the
    second call. Connectors are structure, not data, so this pins a solid line
    and butt caps — the caps matter where segments meet at a right angle.
    """
    return ax.plot(
        xs,
        ys,
        color=color,
        linewidth=linewidth,
        linestyle="-",
        solid_capstyle="butt",
        zorder=zorder,
        **kwargs,
    )[0]


def arrow_props(
    *,
    color: str = EDGE_COLOR,
    linewidth: float = 0.7,
    mutation_scale: float = 5.0,
    arrowstyle: str = "-|>",
    **kwargs: Any,
) -> dict[str, Any]:
    """``arrowprops`` for a directed connector, matching :func:`hairline`.

    ``shrinkA``/``shrinkB`` are zero because callers anchor arrows at a measured
    label or node edge themselves; letting matplotlib shrink from the centre is
    what runs an arrow back through its own label.
    """
    props: dict[str, Any] = {
        "arrowstyle": arrowstyle,
        "mutation_scale": mutation_scale,
        "color": color,
        "linewidth": linewidth,
        "shrinkA": 0,
        "shrinkB": 0,
    }
    props.update(kwargs)
    return props


def draw_node(
    ax: Axes,
    x: float,
    y: float,
    label: str | None = None,
    *,
    inside: bool = True,
    focus: bool = False,
    size: float = NODE_RADIUS * 2,
    fontsize: float | str | None = None,
    label_offset: float = LABEL_OFFSET,
    label_side: str = "right",
    unit: float = 1.0,
    zorder: float = 4.0,
    **kwargs: Any,
) -> Line2D:
    """Draw one network node, plus its label to the right, in the house style.

    The fill carries one bit of membership and nothing else: ``inside`` (the
    default) fills with :data:`NODE_FILL` for a node in the population being
    described, and ``inside=False`` leaves it open (:data:`NODE_OPEN`) for one
    outside it. ``focus=True`` fills with ink, for the single node a figure is
    built around. Every node keeps a black edge so the open fill still reads.

    ``size`` is the marker diameter in points. ``label_offset`` is the gap
    between node centre and label, also in points; ``unit`` converts points to
    data units when the axis carries real units (a year axis), so the label sits
    the same distance from the marker whatever the axis scale. ``label_side``
    flips the label to the left of the marker, for the left end of a paired row.

    Labels are never bold — the focus node is already distinguished by its fill.
    """
    facecolor = INK if focus else (NODE_FILL if inside else NODE_OPEN)
    marker = ax.plot(
        [x],
        [y],
        marker="o",
        markersize=size,
        linestyle="none",
        markerfacecolor=facecolor,
        markeredgecolor=NODE_EDGE,
        markeredgewidth=0.7,
        zorder=zorder,
        **kwargs,
    )[0]
    if label:
        if label_side == "right":
            offset, align = label_offset, "left"
        elif label_side == "left":
            offset, align = -label_offset, "right"
        else:
            raise ValueError(f"label_side must be 'left' or 'right', got {label_side!r}")
        opts: dict[str, Any] = {"ha": align, "va": "center", "color": INK}
        if fontsize is not None:
            opts["fontsize"] = fontsize
        ax.text(x + offset * unit, y, label, zorder=zorder + 1, **opts)
    return marker


def box_node(
    ax: Axes,
    x: float,
    y: float,
    lines: Sequence[str],
    *,
    width: float,
    height: float,
    inside: bool = True,
    focus: bool = False,
    fontsize: float | None = None,
    sub_fontsize: float | None = None,
    zorder: float = 4.0,
    **kwargs: Any,
) -> Any:
    """Draw one network node as a box carrying a name and an attribute.

    The box is :func:`draw_node` with room for a second fact: the first line is
    the node's name, and any line after it is an attribute of that node — an
    institution, a department — set smaller and in :data:`GRAYS` so it reads as
    a subtitle rather than as a second name. The fill still carries membership
    and nothing else, exactly as ``draw_node``: :data:`NODE_FILL` for a node in
    the population being described, :data:`NODE_OPEN` for one outside it, ink
    for the single node a figure is built around. On an ink fill the text flips
    to white, since the alternative is a box with nothing legible in it.

    ``x`` is the **left edge** and ``y`` the **vertical centre**, which is how a
    tree router positions a node: it knows the column a generation starts at and
    the row a person sits on. ``width`` and ``height`` are in data units and
    arrive measured — this function draws a box at the size it is given and
    never asks the renderer how wide the text is, because the layout needs those
    widths before any axes exist.
    """
    from matplotlib.patches import Rectangle

    facecolor = INK if focus else (NODE_FILL if inside else NODE_OPEN)
    patch = Rectangle(
        (x, y - height / 2),
        width,
        height,
        facecolor=facecolor,
        edgecolor=NODE_EDGE,
        linewidth=0.7,
        zorder=zorder,
        **kwargs,
    )
    ax.add_patch(patch)

    base = float(_pyplot().rcParams["font.size"]) if fontsize is None else float(fontsize)
    sub = base - 1.1 if sub_fontsize is None else float(sub_fontsize)
    head_color, sub_color = ("white", GRAYS[3]) if focus else (INK, GRAYS[1])
    step = height / (len(lines) + 1)
    for i, line in enumerate(lines):
        ax.text(
            x + width / 2,
            y + height / 2 - (i + 1) * step,
            line,
            ha="center",
            va="center",
            color=head_color if i == 0 else sub_color,
            fontsize=base if i == 0 else sub,
            zorder=zorder + 1,
        )
    return patch


def flow_band(
    ax: Axes,
    x0: float,
    x1: float,
    *,
    y_left: float,
    y_right: float,
    thickness: float,
    fill: str = BAR_FILL,
    edgecolor: str = INK,
    linewidth: float = 0.4,
    zorder: float = 2.0,
    **kwargs: Any,
) -> Any:
    """Draw one ribbon of a two-column flow diagram.

    A quadrilateral running from ``(x0, y_left)`` to ``(x1, y_right)``, of the
    same ``thickness`` at both ends because both ends count the same people.
    ``y_left`` and ``y_right`` are its **lower** edge, so a caller stacks bands
    by carrying a running offset up each column.

    The edges are straight rather than curved on purpose: a ribbon's job here is
    to be countable, and a sheaf of Béziers reads as a texture. The hairline
    outline is what keeps the lightest fill separable from its neighbour once
    the figure goes through the grayscale proof.
    """
    from matplotlib.patches import Polygon

    band = Polygon(
        [
            (x0, y_left),
            (x1, y_right),
            (x1, y_right + thickness),
            (x0, y_left + thickness),
        ],
        closed=True,
        facecolor=fill,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )
    ax.add_patch(band)
    return band


def node_legend(
    ax: Axes,
    inside_label: str,
    outside_label: str,
    *,
    focus_label: str | None = None,
    loc: str = "lower left",
    size: float = NODE_RADIUS * 2,
    shape: str = "circle",
    **kwargs: Any,
) -> Any:
    """Frameless legend naming what a filled node means and what an open one means.

    Entries follow the same fill vocabulary as :func:`draw_node`, so the legend
    cannot drift from the marks it explains. ``shape="box"`` swaps the circular
    proxies for rectangular ones, for a figure whose nodes are
    :func:`box_node` — a legend of circles under a diagram of boxes invites the
    reader to hunt for a distinction that is not there.
    """
    plt = _pyplot()
    if shape not in ("circle", "box"):
        raise ValueError(f"shape must be 'circle' or 'box', got {shape!r}")
    entries = [(NODE_FILL, inside_label), (NODE_OPEN, outside_label)]
    if focus_label:
        entries.insert(0, (INK, focus_label))
    if shape == "box":
        handles: list[Any] = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=fill,
                edgecolor=NODE_EDGE,
                linewidth=0.7,
                label=text,
            )
            for fill, text in entries
        ]
    else:
        handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor=fill,
                markeredgecolor=NODE_EDGE,
                markeredgewidth=0.7,
                markersize=size,
                label=text,
            )
            for fill, text in entries
        ]
    opts: dict[str, Any] = {
        "frameon": False,
        "loc": loc,
        "ncol": len(handles),
        "handlelength": 1.2,
        "borderaxespad": 0.3,
        "columnspacing": 1.5,
    }
    opts.update(kwargs)
    return ax.legend(handles=handles, **opts)


def separator_rule(
    ax: Axes,
    y: float,
    *,
    xmin: float = 0.0,
    xmax: float = 1.0,
    color: str = RULE_COLOR,
    linewidth: float = 0.6,
    zorder: float = 0.5,
    **kwargs: Any,
) -> Line2D:
    """Hairline separating one band of a grouped figure from the next.

    Lighter than any data mark, drawn under everything, so it groups rows
    without reading as a gridline.
    """
    return ax.axhline(
        y,
        xmin=xmin,
        xmax=xmax,
        color=color,
        linewidth=linewidth,
        linestyle="-",
        zorder=zorder,
        **kwargs,
    )


def show_caption(
    caption: str,
    *,
    notes: str | None = None,
    source: str | None = None,
    number: int | str | None = None,
) -> None:
    """Render the caption block as Markdown under a figure in a notebook.

    The display counterpart of :func:`save_figure`'s caption sidecar, so the
    caption a reader sees in the notebook is built by the same code that writes
    the file next to the artwork.
    """
    from IPython.display import Markdown, display

    display(Markdown(caption_markdown(caption, notes=notes, source=source, number=number)))


def grayscale_preview(png_path: str | Path, out_path: str | Path | None = None) -> Path:
    """Write a grayscale copy of a PNG so the "legible in black and white"
    requirement can be checked by eye. Returns the output path."""
    from PIL import Image

    png_path = Path(png_path)
    out = (
        Path(out_path)
        if out_path is not None
        else png_path.with_name(f"{png_path.stem}.gray{png_path.suffix}")
    )
    with Image.open(png_path) as img:
        img.convert("L").save(out)
    return out
