"""Every import of project code must resolve, including imports inside function bodies.

The `demo/` -> `base_pipeline/` move once left 13 function-local `from demo...` imports
behind. They only failed when the function ran, so 8 eval tests broke and nothing said why.
This test reads every import statement (any depth) in the project packages and checks that its
target module exists and exports the imported names, without executing any function.
"""
import ast
import importlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ("base_pipeline", "clinical_eval", "eval", "whisper_demo")
# `demo` and the top-level `en_pilot` no longer exist; listing them makes any leftover reference
# fail rather than be skipped.
CHECKED = PACKAGES + ("demo", "en_pilot")


def _sources():
    for package in PACKAGES:
        yield from (p for p in sorted((ROOT / package).rglob("*.py")) if "__pycache__" not in p.parts)
    yield ROOT / "app.py"


def _package_of(path):
    """The package a module's relative imports resolve against (its directory)."""
    return ".".join(path.relative_to(ROOT).parts[:-1])


def _imports(path):
    package = _package_of(path)
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, ()
        elif isinstance(node, ast.ImportFrom):
            target = (importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                      if node.level else node.module)
            yield node.lineno, target, tuple(alias.name for alias in node.names)


def _problem(target, names):
    try:
        if importlib.util.find_spec(target) is None:
            return "module not found"
    except ModuleNotFoundError as error:
        return f"module not found ({error.name})"
    module = importlib.import_module(target)
    for name in names:
        if name == "*" or hasattr(module, name):
            continue
        # `from package import submodule` needs no attribute, only a module file.
        if hasattr(module, "__path__") and importlib.util.find_spec(f"{target}.{name}"):
            continue
        return f"{target} has no {name!r}"
    return None


def test_every_project_import_resolves():
    failures = []
    for path in _sources():
        for lineno, target, names in _imports(path):
            if target.split(".")[0] in CHECKED and (problem := _problem(target, names)):
                failures.append(f"{path.relative_to(ROOT)}:{lineno}: import {target} {names or ''} -> {problem}")
    assert not failures, "dangling imports:\n" + "\n".join(failures)
