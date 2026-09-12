# Release Checklist

Use this checklist for the first PyPI release and for later patch releases.

## One-Time PyPI Setup

Configure PyPI trusted publishing for this repository:

- Project name: `citegraph`
- Owner: `yabramuvdi`
- Repository: `citegraph`
- Workflow name: `publish.yml`
- Environment name: `pypi`

Trusted publishing lets GitHub Actions publish without storing a PyPI API token
in the repository.

## Before Tagging

1. Update the version in both `pyproject.toml` and `src/citegraph/__init__.py`.
2. Run local verification:

   ```bash
   python -m pytest tests/test_*.py -vv
   ruff check .
   python -m build
   python -m twine check dist/*
   ```

3. Test a clean wheel install:

   ```bash
   python -m venv /tmp/citegraph-release-check
   /tmp/citegraph-release-check/bin/python -m pip install --upgrade pip
   /tmp/citegraph-release-check/bin/python -m pip install dist/citegraph-*.whl
   /tmp/citegraph-release-check/bin/citegraph --help
   /tmp/citegraph-release-check/bin/python -c "import citegraph; print(citegraph.__version__)"
   ```

4. Commit and push the release-prep change.
5. Confirm CI passes on `main`.

## Publish

Create and push a version tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The `Publish` workflow builds the package, checks metadata, and publishes to
PyPI through trusted publishing.

## After Publishing

Verify the public install path in a fresh environment:

```bash
python -m venv /tmp/citegraph-pypi-check
/tmp/citegraph-pypi-check/bin/python -m pip install --upgrade pip
/tmp/citegraph-pypi-check/bin/python -m pip install citegraph
/tmp/citegraph-pypi-check/bin/citegraph --help
```

## Canonicalization reliability release gate

Install the dev extras and optional graph/plot dependencies used by the suite.
Run the full suite without default test selection and execute the offline gate:

```bash
python -m pytest -o addopts=''
ruff check .
python scripts/evaluate_canonicalization.py \
  --fixtures tests/fixtures/canonicalization \
  --out /private/tmp/citegraph-evaluation.json
```

Require zero false merges on must-separate fixtures and no false splits on
explicitly supported positives. Inspect every changed reversed-input author
partition; report any remaining work-order sensitivity. Check repeats and CSV
roundtrips, and the pipeline tests for cached/fresh and staged/composed equivalence,
registry stability, complete lineage, empty checkpoints, and override validation.
Optional graph/plot tests must execute for a release check. Loopback restrictions
may prevent HTTP-server tests; identify that environment limitation separately
from a code failure and rerun in a permitted environment before claiming a full
release pass.

Follow the copy-only migration in [USER_GUIDE.md](USER_GUIDE.md#rebuild-an-existing-corpus-in-a-copy).
Keep `source_ids.json`, `work_id_collisions.json`, and correction bindings; inspect missing extraction caches
and invalidated enrichment before provider work. Accepted title containments that
introduce negation, different part numbers, or substantive expansions now stay
separate for review; explicit compatible subtitles and supported name variants
remain regression positives. Reports expose unavailable or stale saved evidence
instead of guessing past decisions.

A synthetic fixture pass does not validate representative-corpus accuracy. Freeze
the 150-author/150-work stratified sample and held-out split before tuning. Record
human labels, false merges/splits, independently detected retrieval omissions,
review workload, and migration changes to citation counts and rankings. If human
labels are unavailable, state that the corpus accuracy gate remains unverified.
Do not publish invented precision/recall estimates or imply order invariance.
