"""The README illustrations must stay regenerable without external data or TeX."""
import re
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ['semantic_profiles.svg', 'target_ranking.svg']


def test_readme_image_links():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in IMAGES:
        assert f"docs/figures/{name}" in readme
        ET.parse(ROOT / "docs" / "figures" / name)
    for path in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", readme):
        if not path.startswith(('http:', 'https:')):
            assert (ROOT / path).exists(), f"Missing README target: {path}"


def test_regenerate_synthetic_figures(tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT / "examples" / "make_readme_figures.py"),
                           "--output-dir", str(tmp_path)], cwd=ROOT,
                           capture_output=True, text=True, timeout=90, check=False)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    for name in IMAGES:
        ET.parse(tmp_path / name)
