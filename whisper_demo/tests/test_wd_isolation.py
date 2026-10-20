"""whisper_demo must stand alone: no project imports, and it runs from a copy elsewhere."""
import ast
import shutil
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
FORBIDDEN = {"base_pipeline", "clinical_eval", "eval", "app", "demo"}


def _sources():
    return [p for p in sorted(PKG.rglob("*.py")) if "__pycache__" not in p.parts]


def test_no_import_of_the_rest_of_the_repo():
    bad = []
    for path in _sources():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            elif isinstance(node, ast.ImportFrom) and node.level > 1:
                bad.append(f"{path.name}:{node.lineno} relative import leaves the package")
            bad += [f"{path.name}:{node.lineno} imports {n}" for n in names
                    if n.split(".")[0] in FORBIDDEN]
    assert not bad, bad


def test_runs_from_a_copy_outside_the_repo(tmp_path):
    """Catches `__file__`-relative asset paths that escape the folder."""
    shutil.copytree(PKG, tmp_path / "whisper_demo",
                    ignore=shutil.ignore_patterns("__pycache__", "jobs", "tests"))
    code = (
        "import sys; sys.path.insert(0, sys.argv[1])\n"
        "import whisper_demo.pipeline as p\n"
        "from whisper_demo.extract import normalize_text, extract_entities\n"
        "t = 'Give azithromycin 500 mg once daily 3 days. BP 130 by 80. No penicillin allergy.'\n"
        "e = extract_entities(t, normalize_text(t)['normalized_en'], [])\n"
        "print([d['name'] for d in e['drugs']])\n"
    )
    done = subprocess.run([sys.executable, "-I", "-c", code, str(tmp_path)],
                          cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "azithromycin" in done.stdout.lower()
