"""Tests for multi-section reference extraction (theses with per-chapter bibliographies)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from citegraph.extract_references import (
    extract_references_from_markdown,
    slice_to_references_section,
    split_at_references_header,
    split_reference_sections,
)
from citegraph.llm import GeminiClient
from citegraph.schemas import Reference

CHAPTER_1 = "## Chapter 1\n\nIntro citing Smith (2019).\n\n"
REFS_1 = "## References\n\n- [1] Smith, J., 2019. A paper. Journal A.\n\n"
CHAPTER_2 = "## Chapter 2\n\nMore text citing Jones (2020).\n\n"
REFS_2 = "## References\n\n- [1] Jones, A., 2020. Another paper. Journal B.\n"

THESIS = CHAPTER_1 + REFS_1 + CHAPTER_2 + REFS_2


# ---------------------------------------------------------------------------
# split_reference_sections
# ---------------------------------------------------------------------------
def test_all_sections_captured_and_body_excludes_them() -> None:
    body, sections = split_reference_sections(THESIS)
    assert len(sections) == 2
    assert sections[0] == REFS_1
    assert sections[1] == REFS_2
    assert body == CHAPTER_1 + CHAPTER_2


def test_non_last_section_ends_at_next_same_level_header() -> None:
    text = CHAPTER_1 + REFS_1 + "## APPENDIX A\n\nAppendix text.\n\n" + REFS_2
    body, sections = split_reference_sections(text)
    assert sections[0] == REFS_1
    assert "APPENDIX" not in sections[0]
    assert "APPENDIX" in body


def test_subheader_does_not_terminate_section() -> None:
    refs_with_subheaders = (
        "## References\n\n"
        "### Books\n\n- Smith, J., 2019. A book.\n\n"
        "### Articles\n\n- Jones, A., 2020. An article.\n\n"
    )
    text = CHAPTER_1 + refs_with_subheaders + CHAPTER_2 + REFS_2
    _body, sections = split_reference_sections(text)
    assert len(sections) == 2
    assert sections[0] == refs_with_subheaders


def test_last_section_runs_to_end_of_file() -> None:
    """Unrecognised trailing headers never truncate the last bibliography."""
    trailing = "## Notas finales del autor\n\nUn comentario.\n"
    text = CHAPTER_1 + REFS_2 + trailing
    body, sections = split_reference_sections(text)
    assert sections == [REFS_2 + trailing]
    assert body == CHAPTER_1


def test_no_header_returns_empty_list() -> None:
    text = "## Introduction\n\nNo bibliography here.\n"
    body, sections = split_reference_sections(text)
    assert body == text
    assert sections == []


# ---------------------------------------------------------------------------
# Back-compat wrappers see all sections joined
# ---------------------------------------------------------------------------
def test_split_at_references_header_joins_all_sections() -> None:
    body, refs = split_at_references_header(THESIS)
    assert body == CHAPTER_1 + CHAPTER_2
    assert refs is not None
    assert "Smith, J., 2019" in refs
    assert "Jones, A., 2020" in refs


def test_slice_to_references_section_includes_all_sections() -> None:
    sliced, found = slice_to_references_section(THESIS)
    assert found is True
    assert "Smith, J., 2019" in sliced
    assert "Jones, A., 2020" in sliced
    assert "Intro citing" not in sliced


# ---------------------------------------------------------------------------
# extract_references_from_markdown: one LLM call per section, results merged
# ---------------------------------------------------------------------------
class _RecordingClient(GeminiClient):
    """Answers each prompt from the section text it contains; records prompts."""

    def __init__(self):
        self._client = object()
        self.model = "fake-model"
        from citegraph.config import Settings

        self._settings = Settings(GOOGLE_API_KEY="fake")  # type: ignore[arg-type]
        self._api_key = "fake"
        self.prompts: list[str] = []

    def generate_structured(
        self, *, prompt, response_schema, system_instruction=None, max_output_tokens=None
    ):
        self.prompts.append(prompt)
        refs = []
        if "Smith, J., 2019" in prompt:
            refs.append(
                Reference(
                    Title="A paper", Authors_List=["Smith, J."], Journal="Journal A", Year=2019
                )
            )
        if "Jones, A., 2020" in prompt:
            refs.append(
                Reference(
                    Title="Another paper",
                    Authors_List=["Jones, A."],
                    Journal="Journal B",
                    Year=2020,
                )
            )
        return SimpleNamespace(parsed=refs, text="", candidates=[])


def test_extraction_calls_llm_once_per_section_and_merges(tmp_path: Path) -> None:
    md_path = tmp_path / "thesis.md"
    md_path.write_text(THESIS, encoding="utf-8")
    client = _RecordingClient()

    result = extract_references_from_markdown(md_path, client=client)

    assert len(client.prompts) == 2
    assert "Smith, J., 2019" in client.prompts[0]
    assert "Jones, A., 2020" not in client.prompts[0]
    assert "Jones, A., 2020" in client.prompts[1]
    assert [r.Title for r in result] == ["A paper", "Another paper"]


def test_extraction_single_section_makes_one_call(tmp_path: Path) -> None:
    md_path = tmp_path / "paper.md"
    md_path.write_text(CHAPTER_1 + REFS_1, encoding="utf-8")
    client = _RecordingClient()

    result = extract_references_from_markdown(md_path, client=client)

    assert len(client.prompts) == 1
    assert [r.Title for r in result] == ["A paper"]


# ---------------------------------------------------------------------------
# Back matter terminates the final section (real corpus failure modes)
# ---------------------------------------------------------------------------
# A Gothenburg PhD dissertation ends its last bibliography with the
# department's list of previous theses in the series. It reads exactly like a
# bibliography, so only the header tells the two apart.
GOTHENBURG_BACK_MATTER = (
    "## Previous doctoral theses in the Department of Economics, Gothenburg\n\n"
    "Ostman, Hugo (1911), Norrlands ekonomiska utveckling\n\n"
    "Moritz, Marcus (1911), Den svenska tobaksindustrien\n"
)


def test_last_section_stops_at_previous_theses_back_matter() -> None:
    text = CHAPTER_1 + REFS_1 + GOTHENBURG_BACK_MATTER
    body, sections = split_reference_sections(text)
    assert sections == [REFS_1]
    assert "Norrlands ekonomiska utveckling" in body


def test_last_section_stops_at_appendix() -> None:
    trailing = "## APPENDIX\n\nSurvey instrument.\n"
    text = CHAPTER_1 + REFS_1 + trailing
    _body, sections = split_reference_sections(text)
    assert sections == [REFS_1]


def test_last_section_stops_at_supporting_online_material() -> None:
    trailing = "## Supporting Online Material\n\nMaterials and Methods.\n"
    text = CHAPTER_1 + REFS_1 + trailing
    _body, sections = split_reference_sections(text)
    assert sections == [REFS_1]


def test_unrecognized_trailing_header_still_runs_to_end_of_file() -> None:
    """Fail closed: a running page head inside a bibliography must not truncate it.

    ``## 94 ECONOMIA, Spring 2009`` sits mid-list in a real source paper, with
    ten more references after it.
    """
    running_head = "## 94 ECONOMIA, Spring 2009\n\n- Kim, O., 1984. The Free Rider Problem.\n"
    text = CHAPTER_1 + REFS_1 + running_head
    _body, sections = split_reference_sections(text)
    assert sections == [REFS_1 + running_head]


def test_back_matter_subheader_does_not_terminate_section() -> None:
    """Level still rules: a deeper header never ends a section."""
    refs = "## References\n\n- Smith, J., 2019. A paper.\n\n### Notes\n\n- Jones, A., 2020.\n"
    _body, sections = split_reference_sections(CHAPTER_1 + refs)
    assert sections == [refs]


def test_back_matter_ends_a_non_final_section_too() -> None:
    text = CHAPTER_1 + REFS_1 + "## Acknowledgments\n\nThanks.\n\n" + CHAPTER_2 + REFS_2
    _body, sections = split_reference_sections(text)
    assert sections[0] == REFS_1
