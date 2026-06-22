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
