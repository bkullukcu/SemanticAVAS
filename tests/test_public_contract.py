"""Verify public source syntax and citation consistency without external data."""
import ast
from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_python_files_are_parseable():
    """Catch syntax failures even in example scripts not imported by pytest."""
    paths = [
        file
        for folder in ("src", "examples", "tests")
        for file in (ROOT / folder).rglob("*.py")
    ]
    assert paths, "No public Python sources discovered"
    for file in sorted(paths):
        ast.parse(file.read_text(encoding="utf-8"), filename=str(file))


def test_citation_matches_project_version_and_readme_doi():
    """Guard against release metadata drift across three public entry points."""
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    version = re.search(r"(?m)^version:\s*['\"]?([^\s'\"]+)", citation)
    doi = re.search(r"(?m)^doi:\s*['\"]?([^\s'\"]+)", citation)
    assert version is not None, "CITATION.cff has no version"
    assert doi is not None, "CITATION.cff has no DOI"
    assert version.group(1) == project["version"]
    assert f"https://doi.org/{doi.group(1)}" in readme
