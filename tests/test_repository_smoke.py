"""Smoke test of self-contained public-facing examples and metadata."""
from pathlib import Path
import json
import subprocess
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from semanticavas import __version__

ROOT = Path(__file__).resolve().parents[1]

def test_project_metadata_matches_runtime():
    info = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert info["project"]["version"] == __version__
    assert info["project"]["license"] == "MIT"

def test_synthetic_quickstart_exits_cleanly():
    result = subprocess.run([sys.executable, str(ROOT / "examples" / "quickstart.py")],
        cwd=str(ROOT), text=True, capture_output=True, check=False, timeout=60)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "OK" in result.stdout

def test_no_original_dataset_bundled():
    """Only synthetic fixtures belong in the public repository."""
    assert not (ROOT / "data").exists()
    assert (ROOT / "examples" / "step3_user_data").is_dir()
