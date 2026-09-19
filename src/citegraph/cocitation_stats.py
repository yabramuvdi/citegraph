"""Does a documented tie between two authors show up in how the literature cites them?

A supervision record says that two people are connected socially. An author
co-citation network says which people the field reads together. This module
asks whether the first shows up in the second: for each tie, is the pair
co-cited more often than a *comparable* pair of strangers?

"Comparable" is the whole problem. Advisors and their students are prominent
people, and prominent people are co-cited with everyone, so a tie that beats an
average pair has proved nothing. The null here therefore holds prominence
fixed: each end of a tie is replaced by an author cited by a similar number of
bibliographies, and the tie is scored against the pairs that substitution
produces. What survives is co-citation the two ends' individual visibility does
not already explain.

Two further controls matter on a corpus whose bibliographies were written by
the people being studied:

- ``exclude_self_citations`` drops, for each pair, the bibliographies either
  member wrote. What remains is co-citation by third parties — other people
  putting the two names in one conversation. **The exclusion defines a universe
  rather than filtering one side of a comparison**: the tie's remaining
  bibliographies become the eligible set, and the null pairs are then counted,
  and prominence-matched, inside that same set. Filtering only the tie would
  compare authors who lost most of their chances against authors who lost none
  — on the corpus this was written for, 485 of 564 co-cited authors wrote none
  of the bibliographies at all, while one end of a tie wrote every bibliography
  that cited them. That comparison is not conservative, it is wrong.
- Combined inference needs independent p-values. :func:`fisher_combined_p` is
  retained as an assumption-bound diagnostic, but supervision ties that share
  people are dependent, so it is not confirmatory evidence for overlapping ties.

Nothing here is Gemini- or provider-specific: it reads the finished tables.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd

from citegraph.graph import CitationGraph, author_mentions

__all__ = [
    "TieCocitation",
    "SkippedTie",
    "TieCocitationTest",
    "tie_cocitation_test",
    "fisher_combined_p",
]

#: Columns of :meth:`TieCocitationTest.to_frame`, in order.
TIE_COLUMNS = [
    "source",
    "target",
    "source_name",
    "target_name",
    "observed",
    "null_median",
    "null_mean",
    "p_value",
    "pool_size",
    "n_eligible",
    "n_draws",
]


@dataclass(frozen=True)
class TieCocitation:
    """One tie scored against its degree-matched null.

    ``observed`` is the number of eligible bibliographies citing both ends.
    ``null_median`` / ``null_mean`` summarise the same count over pairs drawn
    from the two ends' prominence strata, and ``p_value`` is the share of those
    draws reaching ``observed`` — counting the observation itself, so it can
    never be reported as zero. ``pool_size`` is the smaller of the two strata:
    a tie scored against a handful of comparable authors carries a p-value with
    very few distinct values, whatever the draw count says.

    ``n_eligible`` is how many bibliographies the tie was scored over. It
    equals the corpus total unless self-citations were excluded, and then it is
    the number that remained — the honest denominator, and the thing to look at
    before reading anything into a tie that came out at zero.
    """

    source: str
    target: str
    source_name: str
    target_name: str
    observed: int
    null_median: float
    null_mean: float
    p_value: float
    pool_size: int
    n_eligible: int
    n_draws: int


@dataclass(frozen=True)
class SkippedTie:
    """A tie the data cannot carry, kept so a caller can report the shortfall."""

    source: str
    target: str
    reason: str


@dataclass(frozen=True)
class TieCocitationTest:
    """Every tie that could be scored, every tie that could not, and the totals."""

    ties: tuple[TieCocitation, ...]
    skipped: tuple[SkippedTie, ...]
    n_citing_works: int
    n_network_authors: int
    n_draws: int
    alpha: float
    exclude_self_citations: bool

    @property
    def n_ties(self) -> int:
        return len(self.ties)

    @property
    def n_significant(self) -> int:
        """Ties whose own permutation p-value falls below ``alpha``."""
        return sum(1 for tie in self.ties if tie.p_value < self.alpha)

    @property
    def expected_significant(self) -> float:
        """How many of that count chance alone would be expected to produce."""
        return self.alpha * self.n_ties

    @property
    def fisher_p_value(self) -> float | None:
        """Fisher's combination of the per-tie p-values.

        This is only interpretable when the supplied ties are independent.
        Overlapping supervision ties violate that assumption, so callers must
        treat this as a diagnostic rather than confirmatory evidence.
        """
        return fisher_combined_p([tie.p_value for tie in self.ties])

    def to_frame(self) -> pd.DataFrame:
        """One row per scored tie, in the order the ties were supplied."""
        if not self.ties:
            return pd.DataFrame(columns=TIE_COLUMNS)
        return pd.DataFrame(
            [{column: getattr(tie, column) for column in TIE_COLUMNS} for tie in self.ties]
        )


def tie_cocitation_test(
    graph: CitationGraph,
    ties: Iterable[tuple[str, str]],
    *,
    min_papers: int = 3,
    n_draws: int = 2000,
    degree_tolerance: float = 0.2,
    exclude_self_citations: bool = False,
    alpha: float = 0.05,
    seed: int = 0,
) -> TieCocitationTest:
    """Score each ``(source, target)`` author pair against a degree-matched null.

    ``min_papers`` is the co-citation network's inclusion threshold and must
    match the network a companion figure draws, or the test and the picture
    describe different populations. ``degree_tolerance`` is the half-width of
    the prominence stratum a substitute is drawn from, as a fraction of the
    replaced author's citing-bibliography count, with a floor of one
    bibliography — at a threshold of three, a 20% band would otherwise admit
    only exact matches.

    Each tie seeds its own generator from its own identity, so the numbers a
    tie gets do not depend on where it sat in ``ties`` or on how many ties came
    with it. A pair given twice, or given both ways round, is scored once.
    """
    graph._require_authors()
    if n_draws < 1:
        raise ValueError(f"n_draws must be at least 1, got {n_draws}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")

    mentions = author_mentions(graph.edges, graph.author_citations, min_papers=min_papers)
    cited_by: dict[str, frozenset[str]] = {
        author: frozenset(group)
        for author, group in mentions.groupby("author_id")["citing_id"]
    }
    universe = sorted(cited_by)

    citing_works = set(graph.edges["citing_id"].unique())
    wrote = graph.author_citations[graph.author_citations["record_id"].isin(citing_works)]
    authored_by: dict[str, frozenset[str]] = {
        author: frozenset(group) for author, group in wrote.groupby("author_id")["record_id"]
    }
    names = graph.authors["display_name"] if "display_name" in graph.authors.columns else None

    def display(author: str) -> str:
        if names is None:
            return author
        return str(names.get(author, author))

    def stratum(
        author: str, exclude: Sequence[str], local_counts: dict[str, int]
    ) -> list[str]:
        width = max(1.0, degree_tolerance * local_counts[author])
        blocked = set(exclude)
        return [
            other
            for other in universe
            if other not in blocked
            and abs(local_counts[other] - local_counts[author]) <= width
        ]

    scored: list[TieCocitation] = []
    skipped: list[SkippedTie] = []
    seen: set[frozenset[str]] = set()

    for source, target in ties:
        if source == target:
            skipped.append(SkippedTie(source, target, "both ends are the same author"))
            continue
        key = frozenset((source, target))
        if key in seen:
            continue
        missing = [end for end in (source, target) if end not in cited_by]
        if missing:
            skipped.append(
                SkippedTie(
                    source,
                    target,
                    f"not in the co-citation network at min_papers={min_papers}: "
                    + ", ".join(missing),
                )
            )
            continue
        seen.add(key)

        # The eligible bibliographies are this tie's universe. Every count
        # below — the tie's, the strata's, the null draws' — is taken inside
        # it, so the tie and its comparison pairs had the same chances.
        if exclude_self_citations:
            eligible = (
                citing_works
                - authored_by.get(source, frozenset())
                - authored_by.get(target, frozenset())
            )
            local_cited = {author: works & eligible for author, works in cited_by.items()}
        else:
            eligible = citing_works
            local_cited = cited_by
        local_counts = {author: len(works) for author, works in local_cited.items()}

        def shared(first: str, second: str, _cited: dict[str, frozenset[str]] = local_cited) -> int:
            return len(_cited[first] & _cited[second])

        source_pool = stratum(source, (source, target), local_counts)
        target_pool = stratum(target, (source, target), local_counts)
        empty_ends = [
            author
            for author, pool in ((source, source_pool), (target, target_pool))
            if not pool
        ]
        if empty_ends:
            skipped.append(
                SkippedTie(
                    source,
                    target,
                    "no degree-matched substitute for: " + ", ".join(empty_ends),
                )
            )
            continue
        if len(set(source_pool) | set(target_pool)) < 2:
            skipped.append(
                SkippedTie(source, target, "no comparable pair of authors to draw a null from")
            )
            continue

        observed = shared(source, target)
        low, high = sorted((source, target))
        rng = random.Random(f"{seed}|{low}|{high}")
        draws: list[int] = []
        # Two draws can collide on one author; redraw rather than score a
        # self-pair, and cap the attempts so a degenerate stratum cannot spin.
        attempts = 0
        while len(draws) < n_draws and attempts < n_draws * 20:
            attempts += 1
            first = rng.choice(source_pool)
            second = rng.choice(target_pool)
            if first == second:
                continue
            draws.append(shared(first, second))
        if not draws:  # pragma: no cover - guarded by the stratum size check
            skipped.append(SkippedTie(source, target, "null sampling produced no valid pair"))
            continue

        at_least = sum(1 for value in draws if value >= observed)
        scored.append(
            TieCocitation(
                source=source,
                target=target,
                source_name=display(source),
                target_name=display(target),
                observed=observed,
                null_median=float(_median(draws)),
                null_mean=float(sum(draws) / len(draws)),
                p_value=(1 + at_least) / (1 + len(draws)),
                pool_size=min(len(source_pool), len(target_pool)),
                n_eligible=len(eligible),
                n_draws=len(draws),
            )
        )

    return TieCocitationTest(
        ties=tuple(scored),
        skipped=tuple(skipped),
        n_citing_works=len(citing_works),
        n_network_authors=len(universe),
        n_draws=n_draws,
        alpha=alpha,
        exclude_self_citations=exclude_self_citations,
    )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


# ----------------------------------------------------------------------
# Combination statistic
# ----------------------------------------------------------------------
def fisher_combined_p(pvalues: Iterable[float]) -> float | None:
    """Combine independent p-values by Fisher's method.

    ``-2 * sum(log p)`` is chi-squared on twice as many degrees of freedom.
    The degrees of freedom are always even, so the survival function has the
    closed form used here and no SciPy dependency is needed for it.

    Returns ``None`` for an empty input. **The independence assumption is
    real**: supervision ties that share an advisor are not independent, and
    this will overstate the evidence for them.
    """
    values = [float(p) for p in pvalues]
    if not values:
        return None
    for value in values:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"p-values must be between 0 and 1, got {value}")
    if any(value == 0.0 for value in values):
        return 0.0
    statistic = -2.0 * sum(math.log(value) for value in values)
    return _chi2_sf_even(statistic, 2 * len(values))


def _chi2_sf_even(statistic: float, df: int) -> float:
    """``P(X > statistic)`` for chi-squared with an even number of df, exactly."""
    half = statistic / 2.0
    term = 1.0
    total = 1.0
    for i in range(1, df // 2):
        term *= half / i
        total += term
    return math.exp(-half) * total
