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
    trailing = "## APPENDIX A\n\nSurvey instrument.\n"
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
