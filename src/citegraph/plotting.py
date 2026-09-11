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

    from citegraph.plotting import econ_style, figure_size, panel_title, save_figure

    with econ_style():
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figure_size("text", ratio=0.45))
        ax_a.bar(years, counts)
        panel_title(ax_a, "A", "Annual counts")
        ...
        save_figure(fig, out_dir / "figures" / "trend",
                    caption="Publication trends", notes="…", source="…")
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
    "econ_rc",
    "econ_style",
    "use_econ_style",
    "figure_size",
    "panel_title",
    "integer_ticks",
    "percent_ticks",
    "reference_line",
    "value_labels",
    "caption_markdown",
    "save_figure",
    "grayscale_preview",
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
) -> list[Path]:
    """Save ``fig`` as ``<stem>.<fmt>`` for each format, plus ``<stem>.caption.md``.

    PDF is the vector file journals ask for (TrueType fonts embedded via
    ``pdf.fonttype = 42``); PNG at ``dpi`` is for Word drafts and slides.
    The caption file keeps title, notes and source next to the artwork so the
    figure itself can stay free of in-plot titles.
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
    return written


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
