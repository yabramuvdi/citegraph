"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_references() -> pd.DataFrame:
    """A tiny DataFrame with two known duplicates and two distinct refs."""
    return pd.DataFrame(
        [
            {
                "Title": "Governing the Commons",
                "Authors_List": ["Elinor Ostrom"],
                "Authors": "Ostrom, E.",
                "Journal": "Cambridge University Press",
                "Year": 1990,
                "citing_id": "p-paper-a",
            },
            {
                "Title": "Governing the commons.",
                "Authors_List": ["E. Ostrom"],
                "Authors": "Elinor Ostrom",
                "Journal": "Cambridge Univ Press",
                "Year": 1990,
                "citing_id": "p-paper-b",
            },
            {
                "Title": "Tragedy of the Commons",
                "Authors_List": ["Garrett Hardin"],
                "Authors": "Hardin, G.",
                "Journal": "Science",
                "Year": 1968,
                "citing_id": "p-paper-a",
            },
            {
                "Title": "Tragedy of the Commons",
                "Authors_List": ["Garrett Hardin"],
                "Authors": "Hardin, Garrett",
                "Journal": "Science",
                "Year": 1968,
                "citing_id": "p-paper-c",
            },
            {
                "Title": "Bowling Alone",
                "Authors_List": ["Robert Putnam"],
                "Authors": "Putnam, R.",
                "Journal": "Journal of Democracy",
                "Year": 1995,
                "citing_id": "p-paper-b",
            },
        ]
    )
