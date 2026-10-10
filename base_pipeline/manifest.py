"""Run manifest: which preset, which flags, which code, written next to the results.

Per-clip provenance (engine, model hashes, actual denoiser) already lives in each arm's
`transcript.json` and `clean_meta.json`; this records what a run was asked to do and the
code state it ran on, so two run folders can be compared without guessing.
"""
import json
import platform
import subprocess
import time
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGES = ("numpy", "soundfile", "faster-whisper", "transformers", "torch", "onnxruntime")


def git_state():
    """{'commit', 'dirty'} for this checkout, or None when it is not a git work tree."""
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              timeout=10, check=True).stdout
    try:
        return {"commit": git("rev-parse", "HEAD").strip(),
                "dirty": bool(git("status", "--porcelain").strip())}
    except (OSError, subprocess.SubprocessError):
        return None


def package_versions():
    found = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def write_manifest(out, *, preset, config, overrides, arms, clips, argv):
    """Write `<out>/manifest.json` and return it."""
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "preset": preset,
        "config": config,
        "overrides": overrides,    # flags typed on top of the preset
        "arms": arms,              # denoiser arms (or the variant) that ran
        "clips": [str(c) for c in clips],
        "argv": argv,
        "git": git_state(),
        "python": platform.python_version(),
        "packages": package_versions(),
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    return manifest
