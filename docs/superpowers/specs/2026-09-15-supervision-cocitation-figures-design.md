# Does the lineage show up in the citation record?

Three figures (11–13) joining paper 4's two layers, plus the tested package code
they need. Figures 1–10 and their alternates keep their stems and their place.

## Why

The eleven institutional figures all describe the 58-author base from the
inside: what these people study, where they trained, who supervised whom. The
rest of paper 4 describes an author co-citation network built from the core
corpus's bibliographies. The two layers never met, although **all 58 carry a
`citegraph` author id and 51 are in that network** — so the join was available
and unused.

That join answers a question the base alone cannot: *do the field's
bibliographies put an advisor and their student in one conversation more often
than they would put two comparable strangers there?* It turns a descriptive
figure set into evidence for a claim.

## What the data supports

Checked against `citegraph_out/` and `institutional/out/` on 2026-09-15.

| Fact | Consequence |
| ---- | ----------- |
| 62 doctoral ties; 20 carry an author id on both ends; 16 have both ends in the co-citation network | Figure 11 tests 16 and names the 4 it cannot |
| 38 of 56 distinct advisors have no author id at all | Coverage is biased towards advisors this literature already reads; stated as a caveat |
| 564 authors over 93 bibliographies at `min_papers=3` | The test's threshold must match the network the manuscript reports |
| 92 of the 93 bibliographies were written by one of the 58 | A self-citation control is mandatory, not optional |
| One end of a tie wrote 39 of the 93; another wrote all 6 that cited them | The control cannot be a one-sided filter — see below |
| 485 of 564 network authors wrote **no** bibliography | The asymmetry that broke the first implementation |
| Cosine co-citation: 0.61 median on supervision ties, 0.13 on all pairs among the same people | Figure 12 has something to show |
| 61% of pairs among the 54 mapped authors are co-cited at least once | Figure 12 must draw a backbone, not the network |
| Doctorate year vs co-citation degree: r = −0.37 over 48 people | Figure 13 is a caveat, not a result |
| `career_positions.csv` still has no year on `affiliation` rows | Career timelines stay ruled out |

## The test

`citegraph.cocitation_stats.tie_cocitation_test`, in the package with unit
tests rather than in the notebook, because it is a manuscript claim and both
paper 4 notebooks are gitignored.

For each tie, count the bibliographies citing both people, then compare against
pairs drawn from the two ends' **prominence strata** — each end replaced by an
author cited by a similar number of bibliographies. Advisors and students are
prominent, and in a network this dense prominent authors are co-cited with
nearly everyone, so a tie that beats an *average* pair has shown nothing.

Two corrections made during the build, both discovered by running the thing
rather than by reasoning about it:

- **The self-citation control cannot filter one side.** The first version
  dropped each pair's own bibliographies from the tie's count and left the null
  pairs counted over everything. That looks conservative and is simply wrong:
  485 of 564 comparison authors wrote no bibliography, so they lose nothing
  while the tie loses most of its chances. It reported **2 of 16** ties
  surviving. The corrected version makes the tie's remaining bibliographies an
  **eligible universe** and re-counts *and re-prominence-matches* the null
  inside it; it reports **7 of 16**. The first number would have been published
  as "the effect does not survive".
- **There is no valid aggregate p-value for these overlapping ties.** Seven
  people appear in more than one tie, so Fisher's method overstates the evidence
  and a binomial count also assumes independent trials. The caption reports the
  per-tie permutation results and treats the significant-tie count as descriptive.

Smaller decisions: p-values count the observation in their own null, so they can
never be reported as zero; each tie seeds its own RNG from its own identity, so
results don't depend on tie order; a tie touching an author below the threshold,
or with at least one endpoint lacking a degree-matched substitute, is skipped
*with a reason* rather than silently scored. `fisher_combined_p` remains available for
independent tie sets and is implemented exactly, so no SciPy dependency is added.

**Result.** 9 of 16 ties beat their own degree-matched null over all
bibliographies and 7 of 16 over third-party bibliographies only, against 0.8
expected at the 5% threshold. Those counts are descriptive, not a combined
significance test. 48% of the co-citation volume on these ties is the pair citing
each other; the finding is what is left after removing it.

## The figures

**Figure 11, `fig11_supervision_cocitation`.** One dumbbell row per tie: filled
marker the observed count, open marker the null median, asterisk where the tie
clears p < 0.05 on its own. Two panels sharing y and x — all bibliographies,
then third-party only — because the attenuation between them *is* a finding.
Rows ordered by panel A's p-value, which is why the most co-cited pair of the
sixteen sits near the bottom.

**Figure 12, `fig12_lineage_on_cocitation`.** The 54 people (51 base authors in
the network plus 3 outside advisors in a scored tie) positioned by
cosine-normalised co-citation, each keeping their 3 strongest ties as a drawn
backbone, with the 16 supervision ties as black arrows on top. Two edge types,
separable in grayscale: light hairline carries the layout, black arrow carries
the claim. Not every arrow is short, and that is the point of drawing it —
`Croson → Candelo` crosses the map and is one of the ties figure 11 cannot
distinguish from chance.

**Figure 13, `fig13_prominence_and_academic_age`.** Doctorate year against
co-citation degree, 48 people, OLS fit, four names. The notebook's only scatter.
A caveat figure: part of the spread in prominence across this list is time in
the field.

## Code added

- **`src/citegraph/cocitation_stats.py`** — the test, its result types, and the
  two combination statistics. 27 unit tests.
- **`author_mentions()` in `graph.py`** — the projection's atom, one row per
  bibliography-cites-author, extracted so `author_cocitation_network` and the
  tie test cannot drift. A test asserts a tie's observed count equals the
  co-citation edge weight a companion figure draws.
- **`place_node_labels()` in `plotting.py`** — collision-free labels for a
  node-link drawing, shared rather than re-derived per notebook. Takes positions
  for every node but labels for a subset, because an unlabelled node is still an
  obstacle. Measures each candidate **without** its leader line: an annotation
  that owns an arrow reports the arrow inside its extent, so a far anchor always
  measures as a collision and is never chosen — a bug that silently disabled the
  outer anchors until it was found. Where no anchor is clean the least
  overlapping one wins, never the preferred one, or a clump's names all stack in
  one place. 9 unit tests.

## Verification

- 662 tests pass; `ruff` clean.
- The notebook executed three times in separate processes under
  `PYTHONHASHSEED` 1, 98765 and 424242: **all 36 PNGs byte-identical**.
- Grayscale proofs written for all three new figures; both edge types in figure
  12 remain separable.

## Ruled out, with the evidence

- **Institution-level training→placement flow.** 44 PhD institutions and 43
  affiliation institutions over 56 people, HHI 0.03 both sides, only 4 people
  still at the institution that trained them. Too sparse for ribbons. The
  interesting part is a placement asymmetry (three Bogotá institutions awarded
  3 doctorates and employ 14), which is a paired bar chart, not a flow, and is
  a separate figure nobody has asked for yet.
- **Drawing the full 564-author network** behind figure 12. 61% density among
  the mapped authors alone; the backbone is what makes a layout resolve.
- **Encoding significance on figure 12's arrows.** Figure 11 carries it; a
  second encoding on the map would invite reading arrow weight as tie strength.
- **Academic sibling pairs as a fourth figure.** Only 8 are testable and the
  lifts are carried by pairs with 3 co-citations each. The numbers are in the
  session record if it is ever worth revisiting.
