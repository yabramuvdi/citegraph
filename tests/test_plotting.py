"""Tests for the economics-journal matplotlib style helpers in ``citegraph.plotting``."""

from __future__ import annotations

from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

from citegraph import plotting  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ----------------------------------------------------------------------
# rcParams / style context
# ----------------------------------------------------------------------


def test_econ_rc_is_serif_ink_on_white_with_two_spines() -> None:
    rc = plotting.econ_rc()

    assert rc["font.family"] == "serif"
    assert rc["font.serif"][0] == "Times New Roman"
    assert rc["axes.spines.top"] is False
    assert rc["axes.spines.right"] is False
    assert rc["axes.spines.left"] is True
    assert rc["axes.spines.bottom"] is True
    assert rc["axes.grid"] is False
    assert rc["axes.edgecolor"] == plotting.INK
    assert rc["figure.facecolor"] == "white"
    assert rc["savefig.facecolor"] == "white"
    # Journals reject Type 3 fonts; 42 embeds TrueType.
    assert rc["pdf.fonttype"] == 42
    assert rc["ps.fonttype"] == 42
    # No bold anywhere: captions carry emphasis, not the figure.
    assert rc["axes.titleweight"] == "normal"
    assert rc["axes.labelweight"] == "normal"
    assert rc["font.weight"] == "normal"


def test_econ_rc_font_sizes_are_print_sized() -> None:
    rc = plotting.econ_rc()

    for key in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize",
                "ytick.labelsize", "legend.fontsize"):
        assert 7 <= float(rc[key]) <= 11, key

    scaled = plotting.econ_rc(base_size=12)
    assert scaled["font.size"] == 12
    assert float(scaled["xtick.labelsize"]) < 12


def test_mono_palette_cycles_black_first_with_distinct_linestyles() -> None:
    rc = plotting.econ_rc(palette="mono")
    cycle = list(rc["axes.prop_cycle"])

    assert cycle[0]["color"].lower() == plotting.INK
    linestyles = [step["linestyle"] for step in cycle]
    assert len(set(linestyles)) == len(linestyles), "each grayscale series needs its own dash"


def test_color_palette_is_okabe_ito_after_black() -> None:
    rc = plotting.econ_rc(palette="color")
    colors = [step["color"].upper() for step in rc["axes.prop_cycle"]]

    assert colors[0] == plotting.INK.upper()
    assert colors[1:] == [c.upper() for c in plotting.OKABE_ITO]
    # Colour never travels alone: line style still varies so grayscale print survives.
    assert len({step["linestyle"] for step in rc["axes.prop_cycle"]}) > 1


def test_unknown_palette_raises() -> None:
    with pytest.raises(ValueError, match="palette"):
        plotting.econ_rc(palette="rainbow")


def test_econ_style_context_is_scoped() -> None:
    before = plt.rcParams["font.family"]
    with plotting.econ_style():
        assert plt.rcParams["font.family"] == ["serif"]
        assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["font.family"] == before


def test_econ_style_accepts_overrides() -> None:
    with plotting.econ_style(overrides={"axes.grid": True}):
        assert plt.rcParams["axes.grid"] is True


# ----------------------------------------------------------------------
# Sizes
# ----------------------------------------------------------------------


def test_figure_size_presets_match_manuscript_widths() -> None:
    assert plotting.figure_size("text") == pytest.approx((6.5, 6.5 * 0.6))
    assert plotting.figure_size("column")[0] == pytest.approx(3.25)
    assert plotting.figure_size("journal")[0] == pytest.approx(4.75)
    assert plotting.figure_size(5.0, ratio=0.5) == pytest.approx((5.0, 2.5))
    assert plotting.figure_size("text", height=4.0) == (6.5, 4.0)
    with pytest.raises(ValueError, match="width"):
        plotting.figure_size("poster")


# ----------------------------------------------------------------------
# Axis helpers
# ----------------------------------------------------------------------


def test_panel_title_uses_aer_wording() -> None:
    fig, ax = plt.subplots()
    plotting.panel_title(ax, "A", "Annual counts")
    assert ax.get_title() == "Panel A. Annual counts"

    plotting.panel_title(ax, "B")
    assert ax.get_title() == "Panel B"


def test_integer_and_percent_ticks() -> None:
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    fig, ax = plt.subplots()
    ax.bar([0, 1], [1, 2])
    plotting.integer_ticks(ax, axis="y")
    assert isinstance(ax.yaxis.get_major_locator(), MaxNLocator)
    assert all(float(t).is_integer() for t in ax.get_yticks())

    plotting.percent_ticks(ax, axis="y", xmax=100)
    assert isinstance(ax.yaxis.get_major_formatter(), PercentFormatter)
    assert ax.yaxis.get_major_formatter()(50) == "50%"

    plotting.percent_ticks(ax, axis="x", xmax=1, decimals=1)
    assert ax.xaxis.get_major_formatter()(0.5) == "50.0%"

    with pytest.raises(ValueError, match="axis"):
        plotting.integer_ticks(ax, axis="z")


def test_reference_line_is_dashed_ink_and_labelled() -> None:
    fig, ax = plt.subplots()
    ax.bar(["a", "b"], [10, 30])
    line = plotting.reference_line(ax, 20, label="All periods: 20%")

    assert line.get_linestyle() == "--"
    assert line.get_color() == plotting.INK
    assert line.get_ydata()[0] == 20
    texts = [t.get_text() for t in ax.texts]
    assert "All periods: 20%" in texts

    vline = plotting.reference_line(ax, 0.5, orientation="v")
    assert vline.get_xdata()[0] == 0.5

    n_texts = len(ax.texts)
    legend_line = plotting.reference_line(ax, 25, label="Overall: 25%", in_legend=True)
    assert legend_line.get_label() == "Overall: 25%"
    assert len(ax.texts) == n_texts, "in_legend must not draw free text"
    assert "Overall: 25%" in [t.get_text() for t in ax.legend().get_texts()]
    with pytest.raises(ValueError, match="orientation"):
        plotting.reference_line(ax, 1, orientation="diag")


def test_value_labels_are_regular_weight_ink() -> None:
    fig, ax = plt.subplots()
    bars = ax.bar(["a", "b"], [1234, 5])
    labels = plotting.value_labels(ax, bars, fmt="{:,.0f}")

    assert [t.get_text() for t in labels] == ["1,234", "5"]
    assert all(t.get_fontweight() in ("normal", 400) for t in labels)
    assert all(t.get_color() == plotting.INK for t in labels)


# ----------------------------------------------------------------------
# Saving + captions
# ----------------------------------------------------------------------


def test_caption_markdown_layout() -> None:
    text = plotting.caption_markdown(
        "Publication trends",
        notes="Bars show annual counts.",
        source="citegraph output.",
        number=1,
    )
    lines = text.splitlines()
    assert lines[0] == "**Figure 1. Publication trends**"
    assert "*Notes:* Bars show annual counts." in text
    assert "*Source:* citegraph output." in text
    # Notes precede source, as the AER style guide requires.
    assert text.index("*Notes:*") < text.index("*Source:*")

    # Unnumbered: the title stands alone so the figure can be numbered when arranged.
    bare = plotting.caption_markdown("Only a title")
    assert bare == "**Only a title**"


def test_save_figure_writes_vector_raster_and_caption(tmp_path: Path) -> None:
    with plotting.econ_style():
        fig, ax = plt.subplots(figsize=plotting.figure_size("column"))
        ax.bar([0, 1], [1, 2])
        written = plotting.save_figure(
            fig,
            tmp_path / "figs" / "trend",
            caption="Publication trends",
            notes="Test notes.",
            source="Test source.",
            number=3,
        )

    names = sorted(p.name for p in written)
    assert names == ["trend.caption.md", "trend.pdf", "trend.png"]
    assert all(p.exists() and p.stat().st_size > 0 for p in written)
    caption = (tmp_path / "figs" / "trend.caption.md").read_text(encoding="utf-8")
    assert caption.startswith("**Figure 3. Publication trends**")
    assert "*Notes:* Test notes." in caption


def test_save_figure_without_caption_skips_caption_file(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    written = plotting.save_figure(fig, tmp_path / "plain", formats=("png",), dpi=72)

    assert [p.name for p in written] == ["plain.png"]
    assert not (tmp_path / "plain.caption.md").exists()


def test_grayscale_preview_writes_single_channel_png(tmp_path: Path) -> None:
    from PIL import Image

    fig, ax = plt.subplots()
    ax.bar([0, 1], [1, 2], color=["#0072B2", "#D55E00"])
    (png,) = plotting.save_figure(fig, tmp_path / "color", formats=("png",), dpi=72)

    gray = plotting.grayscale_preview(png)
    assert gray == tmp_path / "color.gray.png"
    with Image.open(gray) as img:
        assert img.mode == "L"
