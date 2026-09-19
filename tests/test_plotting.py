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
    assert names == ["trend.caption.md", "trend.gray.png", "trend.pdf", "trend.png"]
    assert all(p.exists() and p.stat().st_size > 0 for p in written)
    # The B&W proof lands in gray/, not beside the artwork handed to a coauthor.
    assert (tmp_path / "figs" / "gray" / "trend.gray.png").exists()
    caption = (tmp_path / "figs" / "trend.caption.md").read_text(encoding="utf-8")
    assert caption.startswith("**Figure 3. Publication trends**")
    assert "*Notes:* Test notes." in caption


def test_save_figure_without_caption_skips_caption_file(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    written = plotting.save_figure(
        fig, tmp_path / "plain", formats=("png",), dpi=72, grayscale=False
    )

    assert [p.name for p in written] == ["plain.png"]
    assert not (tmp_path / "plain.caption.md").exists()
    assert not (tmp_path / "gray").exists()


def test_grayscale_preview_writes_single_channel_png(tmp_path: Path) -> None:
    from PIL import Image

    fig, ax = plt.subplots()
    ax.bar([0, 1], [1, 2], color=["#0072B2", "#D55E00"])
    (png,) = plotting.save_figure(
        fig, tmp_path / "color", formats=("png",), dpi=72, grayscale=False
    )

    gray = plotting.grayscale_preview(png)
    assert gray == tmp_path / "color.gray.png"
    with Image.open(gray) as img:
        assert img.mode == "L"


# ----------------------------------------------------------------------
# Shared marks: bars and the network vocabulary
# ----------------------------------------------------------------------


def test_bar_helpers_fill_gray_because_the_prop_cycle_cannot() -> None:
    """``ax.bar`` takes its colour from the property cycle, whose first entry is
    ink, so bare bars come out solid black under ``econ_style`` however
    ``patch.facecolor`` is set. The helpers are the only way to get the
    documented gray-fill-black-edge bar, which is why callers must use them."""
    from matplotlib.colors import to_hex

    with plotting.econ_style(overrides={"patch.facecolor": plotting.BAR_FILL}):
        fig, ax = plt.subplots()
        bare = ax.bar([0, 1], [1, 2])
        assert to_hex(bare[0].get_facecolor()) == plotting.INK.lower()

        vertical = plotting.bar(ax, [0, 1], [1, 2])
        horizontal = plotting.barh(ax, [0, 1], [1, 2])

    for container in (vertical, horizontal):
        assert to_hex(container[0].get_facecolor()) == plotting.BAR_FILL.lower()
        assert to_hex(container[0].get_edgecolor()) == plotting.INK.lower()


def test_bar_helpers_let_an_explicit_colour_win() -> None:
    """Stacked and multi-series bars pass their own shade; the default must not
    override it."""
    from matplotlib.colors import to_hex

    fig, ax = plt.subplots()
    stacked = plotting.bar(ax, [0, 1], [1, 2], color=plotting.GRAYS[3], hatch="////")
    assert to_hex(stacked[0].get_facecolor()) == plotting.GRAYS[3].lower()
    assert stacked[0].get_hatch() == "////"


def test_year_ticks_thin_long_spans_and_stay_horizontal() -> None:
    fig, ax = plt.subplots()
    plotting.year_ticks(ax, 1993, 2024)
    ticks = list(ax.get_xticks())
    assert ticks == [1995, 2000, 2005, 2010, 2015, 2020]
    assert ax.get_xlim() == (1992.3, 2024.7)

    fig2, ax2 = plt.subplots()
    plotting.year_ticks(ax2, 2010, 2020)
    assert list(ax2.get_xticks()) == [2010, 2012, 2014, 2016, 2018, 2020]


def test_draw_node_fill_carries_membership_and_never_bolds_the_label() -> None:
    from matplotlib.colors import to_hex

    fig, ax = plt.subplots()
    inside = plotting.draw_node(ax, 0, 0, "Inside", inside=True)
    outside = plotting.draw_node(ax, 0, 1, "Outside", inside=False)
    focus = plotting.draw_node(ax, 0, 2, "Focus", focus=True)

    assert to_hex(inside.get_markerfacecolor()) == plotting.NODE_FILL.lower()
    assert to_hex(outside.get_markerfacecolor()) == to_hex(plotting.NODE_OPEN)
    assert to_hex(focus.get_markerfacecolor()) == plotting.INK.lower()
    for marker in (inside, outside, focus):
        assert to_hex(marker.get_markeredgecolor()) == plotting.NODE_EDGE.lower()
    # The focus node is distinguished by fill, never by weight.
    assert all(text.get_fontweight() == "normal" for text in ax.texts)


def test_draw_node_offsets_its_label_in_data_units_via_unit() -> None:
    """On a year axis one point is not one year, so the caller passes the
    conversion and the label keeps the same visual gap at any scale."""
    fig, ax = plt.subplots()
    plotting.draw_node(ax, 2000, 0, "Name", label_offset=6.0, unit=0.5)
    assert ax.texts[0].get_position() == (2003.0, 0)


def test_node_legend_reuses_the_draw_node_fills() -> None:
    from matplotlib.colors import to_hex

    fig, ax = plt.subplots()
    legend = plotting.node_legend(ax, "In the list", "Outside it", focus_label="Focus")
    fills = [to_hex(h.get_markerfacecolor()) for h in legend.legend_handles]
    assert fills == [plotting.INK, plotting.NODE_FILL, to_hex(plotting.NODE_OPEN)]
    assert legend.get_frame_on() is False


def test_network_axes_removes_all_chrome() -> None:
    fig, ax = plt.subplots()
    plotting.network_axes(ax)
    assert not ax.axison
    assert ax.get_aspect() == 1.0


def test_separator_rule_is_lighter_than_any_data_mark_and_sits_underneath() -> None:
    from matplotlib.colors import to_hex

    fig, ax = plt.subplots()
    rule = plotting.separator_rule(ax, 0.5)
    assert to_hex(rule.get_color()) == plotting.RULE_COLOR.lower()
    assert rule.get_zorder() < 1.0
    assert rule.get_linestyle() == "-"


def test_junction_dot_is_ink_on_the_connector_not_a_node() -> None:
    """A junction is punctuation on a hairline. Drawn anywhere near node size it
    reads as an unlabelled person, which is the opposite of clarifying."""
    from matplotlib.colors import to_hex

    fig, ax = plt.subplots()
    dot = plotting.junction_dot(ax, 3.0, 4.0)
    assert dot.get_xydata().tolist() == [[3.0, 4.0]]
    assert to_hex(dot.get_markerfacecolor()) == plotting.EDGE_COLOR.lower()
    assert dot.get_markersize() == pytest.approx(plotting.JUNCTION_RADIUS * 2)
    assert plotting.JUNCTION_RADIUS < plotting.NODE_RADIUS / 2


def test_junction_dot_sits_above_the_connectors_it_punctuates() -> None:
    """It has to cover the hairlines meeting under it, and stay below the nodes."""
    fig, ax = plt.subplots()
    line = plotting.hairline(ax, [0, 1], [0, 0])
    dot = plotting.junction_dot(ax, 0.5, 0.0)
    node = plotting.draw_node(ax, 1.0, 0.0)
    assert line.get_zorder() < dot.get_zorder() < node.get_zorder()


def test_draw_node_can_hang_its_label_on_the_left() -> None:
    """The left end of a paired row needs its label outside the pair, not
    running back across the connector."""
    fig, ax = plt.subplots()
    plotting.draw_node(ax, 10, 0, "Advisor", label_side="left", label_offset=6.0)
    text = ax.texts[0]
    assert text.get_position() == (4.0, 0)
    assert text.get_ha() == "right"

    with pytest.raises(ValueError, match="label_side"):
        plotting.draw_node(ax, 0, 0, "x", label_side="above")


# ----------------------------------------------------------------------
# Box nodes and flow bands
# ----------------------------------------------------------------------


def test_box_node_fill_carries_membership_exactly_as_draw_node_does() -> None:
    from matplotlib.colors import to_hex

    fig, ax = plt.subplots()
    inside = plotting.box_node(ax, 0, 0, ["Inside"], width=10, height=4, inside=True)
    outside = plotting.box_node(ax, 0, 5, ["Outside"], width=10, height=4, inside=False)
    focus = plotting.box_node(ax, 0, 10, ["Focus"], width=10, height=4, focus=True)

    assert to_hex(inside.get_facecolor()) == plotting.BOX_FILL.lower()
    assert to_hex(outside.get_facecolor()) == to_hex(plotting.NODE_OPEN)
    assert to_hex(focus.get_facecolor()) == plotting.INK.lower()
    for patch in (inside, outside, focus):
        assert to_hex(patch.get_edgecolor()) == plotting.NODE_EDGE.lower()
    assert all(text.get_fontweight() == "normal" for text in ax.texts)


def test_box_node_anchors_on_its_left_edge_at_the_given_size() -> None:
    """The tree router positions a node by its left edge and centres it
    vertically, so that is what the box has to honour."""
    fig, ax = plt.subplots()
    patch = plotting.box_node(ax, 100, 50, ["Name"], width=30, height=8)

    assert patch.get_xy() == (100, 46.0)
    assert patch.get_width() == 30
    assert patch.get_height() == 8


def test_box_node_stacks_two_lines_symmetrically_about_the_centre() -> None:
    fig, ax = plt.subplots()
    plotting.box_node(ax, 0, 0, ["Victoria Chick", "UCL"], width=40, height=12)

    name, institution = ax.texts
    assert name.get_position() == (20.0, 2.0)
    assert institution.get_position() == (20.0, -2.0)
    assert name.get_ha() == "center"


def test_box_node_sets_the_second_line_smaller_and_grayer_than_the_name() -> None:
    from matplotlib.colors import to_hex

    fig, ax = plt.subplots()
    plotting.box_node(ax, 0, 0, ["Name", "Institution"], width=40, height=12, fontsize=8.0)

    name, institution = ax.texts
    assert institution.get_fontsize() < name.get_fontsize()
    assert to_hex(institution.get_color()) == plotting.GRAYS[1].lower()
    assert to_hex(name.get_color()) == plotting.INK.lower()


def test_box_node_on_an_ink_fill_puts_light_text_on_the_dark_box() -> None:
    """A focus box is filled with ink, so black text on it would be invisible."""
    from matplotlib.colors import to_hex

    fig, ax = plt.subplots()
    plotting.box_node(ax, 0, 0, ["Name", "Institution"], width=40, height=12, focus=True)

    assert to_hex(ax.texts[0].get_color()) == "#ffffff"


def test_flow_band_closes_a_quadrilateral_between_the_two_stacks() -> None:
    fig, ax = plt.subplots()
    band = plotting.flow_band(ax, 0.0, 10.0, y_left=2.0, y_right=5.0, thickness=3.0)

    corners = [tuple(p) for p in band.get_xy()[:4]]
    assert corners == [(0.0, 2.0), (10.0, 5.0), (10.0, 8.0), (0.0, 5.0)]


def test_flow_band_keeps_a_hairline_edge_so_the_lightest_fill_still_reads() -> None:
    from matplotlib.colors import to_hex

    fig, ax = plt.subplots()
    band = plotting.flow_band(ax, 0, 1, y_left=0, y_right=0, thickness=1, fill=plotting.GRAYS[3])

    assert to_hex(band.get_facecolor()) == plotting.GRAYS[3].lower()
    assert to_hex(band.get_edgecolor()) == plotting.INK.lower()
    assert band.get_linewidth() > 0


def test_node_legend_draws_box_proxies_when_the_marks_are_boxes() -> None:
    from matplotlib.colors import to_hex

    fig, ax = plt.subplots()
    legend = plotting.node_legend(ax, "In the list", "Outside it", shape="box")

    fills = [to_hex(h.get_facecolor()) for h in legend.legend_handles]
    assert fills == [plotting.BOX_FILL.lower(), to_hex(plotting.NODE_OPEN)]
    assert legend.get_frame_on() is False


def test_node_legend_rejects_a_shape_it_has_no_mark_for() -> None:
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="shape"):
        plotting.node_legend(ax, "In", "Out", shape="triangle")


def test_box_fill_is_light_enough_to_keep_the_text_inside_it_black() -> None:
    """A box is a hundred times the area of a marker, so it takes the light end
    of the ramp where NODE_FILL takes the dark one. The failure this prevents is
    concrete: at NODE_FILL the subtitle colour and the fill were the same gray,
    and every institution line inside a filled box was invisible."""
    assert plotting.BOX_FILL in plotting.GRAYS
    assert plotting.GRAYS.index(plotting.BOX_FILL) > plotting.GRAYS.index(plotting.NODE_FILL)


def test_box_node_subtitle_stays_legible_on_both_box_fills() -> None:
    from matplotlib.colors import to_hex

    fig, ax = plt.subplots()
    plotting.box_node(ax, 0, 0, ["Name", "Institution"], width=40, height=12, inside=True)
    plotting.box_node(ax, 0, 20, ["Name", "Institution"], width=40, height=12, inside=False)

    inside_sub, outside_sub = ax.texts[1], ax.texts[3]
    assert to_hex(inside_sub.get_color()) == to_hex(outside_sub.get_color())
    assert to_hex(inside_sub.get_color()) != to_hex(plotting.BOX_FILL)


# ----------------------------------------------------------------------
# Collision-free labels for a node-link drawing
# ----------------------------------------------------------------------
def _labelled_axes(positions, labels, **kwargs):
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    placed = plotting.place_node_labels(ax, labels, positions, **kwargs)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = {key: text.get_window_extent(renderer=renderer) for key, text in placed.items()}
    return ax, placed, boxes


def test_place_node_labels_labels_every_node_it_was_asked_to() -> None:
    positions = {"a": (-0.5, 0.0), "b": (0.5, 0.0), "c": (0.0, 0.5)}
    _, placed, _ = _labelled_axes(positions, {"a": "Alpha", "b": "Beta", "c": "Gamma"})

    assert set(placed) == {"a", "b", "c"}
    assert {text.get_text() for text in placed.values()} == {"Alpha", "Beta", "Gamma"}


def test_place_node_labels_only_labels_the_keys_given() -> None:
    """Unlabelled nodes still take part: they are obstacles, not absentees."""
    positions = {"a": (-0.5, 0.0), "b": (0.5, 0.0), "quiet": (0.0, 0.0)}
    _, placed, _ = _labelled_axes(positions, {"a": "Alpha", "b": "Beta"})

    assert set(placed) == {"a", "b"}


def test_place_node_labels_separates_names_of_neighbouring_nodes() -> None:
    """Two nodes close enough to share a preferred anchor must not overlap."""
    positions = {"a": (0.0, 0.0), "b": (0.02, 0.0)}
    _, _, boxes = _labelled_axes(positions, {"a": "Anderson", "b": "Bernal"})

    assert not boxes["a"].overlaps(boxes["b"])


def test_place_node_labels_keeps_a_label_off_a_node_marker() -> None:
    """A big marker directly above a node must push its label elsewhere."""
    positions = {"a": (0.0, 0.0), "big": (0.0, 0.06)}
    ax, placed, boxes = _labelled_axes(
        positions, {"a": "Alpha"}, node_sizes={"a": 6.0, "big": 40.0}
    )

    centre = ax.transData.transform(positions["big"])
    box = boxes["a"]
    nearest_x = min(max(centre[0], box.x0), box.x1)
    nearest_y = min(max(centre[1], box.y0), box.y1)
    assert ((centre[0] - nearest_x) ** 2 + (centre[1] - nearest_y) ** 2) ** 0.5 >= 20.0


def test_place_node_labels_draws_a_leader_only_when_the_label_travelled() -> None:
    """A name far from every marker belongs to none of them without a leader."""
    # More names than the near ring of anchors can hold, so the overflow is
    # pushed out to the far ring and has to be led back to its node.
    crowd = {f"n{i:02d}": (0.0, i * 0.002) for i in range(9)}
    _, placed, _ = _labelled_axes(
        crowd, {key: "Widename" for key in crowd}, reaches=(6.0, 40.0), leader_beyond=10.0
    )
    leaders = {key for key, text in placed.items() if text.arrow_patch is not None}

    assert leaders, "the crowded names had to travel but got no leader"
    assert len(leaders) < len(crowd), "a name that kept its preferred anchor needs no leader"
    for key, text in placed.items():
        travelled = max(abs(text.xyann[0]), abs(text.xyann[1])) > 10.0
        assert (key in leaders) == travelled


def test_place_node_labels_spreads_a_crowd_no_anchor_can_satisfy() -> None:
    """When nowhere is clean the least-bad anchor wins, not the preferred one —
    otherwise a whole clump's names stack in a single spot."""
    crowd = {f"n{i:02d}": (0.0, i * 0.002) for i in range(9)}
    _, placed, _ = _labelled_axes(crowd, {key: "Verylongsurname" for key in crowd})

    anchors = [text.xyann for text in placed.values()]
    assert len(set(anchors)) > len(anchors) / 2


def test_place_node_labels_follows_the_order_it_is_given() -> None:
    """Placement is priority-ordered, so the caller decides which name gets the
    preferred anchor — and the drawing is identical from run to run."""
    positions = {"a": (0.0, 0.0), "b": (0.015, 0.0)}
    labels = {"a": "Alpha", "b": "Beta"}
    _, first, first_boxes = _labelled_axes(positions, labels, order=["a", "b"])
    _, second, second_boxes = _labelled_axes(positions, labels, order=["b", "a"])

    assert first["a"].xyann != second["a"].xyann
    assert first_boxes["a"].y0 > first_boxes["b"].y0
    assert second_boxes["b"].y0 > second_boxes["a"].y0


def test_place_node_labels_is_reproducible_for_one_input() -> None:
    positions = {"a": (0.0, 0.0), "b": (0.02, 0.0), "c": (-0.02, 0.01)}
    labels = {"a": "Alpha", "b": "Beta", "c": "Gamma"}
    _, first, _ = _labelled_axes(positions, labels)
    _, second, _ = _labelled_axes(positions, labels)

    assert {k: v.xyann for k, v in first.items()} == {k: v.xyann for k, v in second.items()}


def test_place_node_labels_rejects_a_label_for_a_node_it_has_no_position_for() -> None:
    fig, ax = plt.subplots()
    with pytest.raises(KeyError, match="ghost"):
        plotting.place_node_labels(ax, {"ghost": "Nobody"}, {"a": (0.0, 0.0)})
