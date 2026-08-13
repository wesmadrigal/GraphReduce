# Releasing GraphReduce

PyPI does not provide a separate release-notes field. GraphReduce therefore
keeps durable release notes in `CHANGELOG.md`, appends that file to the package
description shown on PyPI, links it in the project metadata, and uses the
matching changelog entry as the GitHub Release description.

## One-time setup

1. In the GitHub repository settings, create an environment named `pypi` and
   require a maintainer's approval before deployment.
2. In the GraphReduce project's PyPI publishing settings, add a GitHub Trusted
   Publisher with:
   - Owner: `wesmadrigal`
   - Repository: `graphreduce`
   - Workflow: `release.yml`
   - Environment: `pypi`

No long-lived PyPI API token is needed.

## Prepare a release

1. Update the version in `setup.py` and `graphreduce/__init__.py`.
2. Add a dated entry to `CHANGELOG.md`.
3. Run the test suite.
4. Build and check the distributions locally:

   ```bash
   python -m pip install --upgrade build twine
   python -m build --outdir release-dist
   python -m twine check release-dist/*
   ```

5. Commit the release changes and push them to the default branch.

## Publish

1. On GitHub, draft a new release whose tag is the package version prefixed
   with `v`, such as `v1.9.17`.
2. Use **Generate release notes**, then replace or supplement the generated
   summary with the matching `CHANGELOG.md` entry.
3. Publish the GitHub Release.
4. Approve the `pypi` environment deployment after the build job succeeds.

Publishing the GitHub Release starts `.github/workflows/release.yml`. The
workflow verifies the tag/version pair, builds a fresh wheel and source
distribution, validates their metadata, and publishes them to PyPI using
Trusted Publishing.
