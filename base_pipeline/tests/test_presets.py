"""Spec: named presets are data, the CLI resolves them, and every run records what it did."""
import json

import pytest

from base_pipeline import decode_presets, manifest, presets, run
from base_pipeline.denoise import BackendId

# The pipeline's historical command-line defaults, written out: `baseline` must stay equal to
# them, or "python -m base_pipeline.run clip.wav" silently changes meaning.
HISTORICAL_DEFAULTS = {
    "denoiser": "none", "model": "medasr", "decode": "greedy", "second_pass": "auto",
    "use_llm": False, "validate_terms": "auto", "context_repair": False,
    "variant": None, "snr_threshold": 15.0, "whisper_model": "small-int8",
}


def _parse(*argv):
    return run.build_parser().parse_args(["clip.wav", *argv])


def test_baseline_is_exactly_the_historical_default():
    assert presets.resolve("baseline") == HISTORICAL_DEFAULTS
    assert presets.DEFAULT == "baseline"


@pytest.mark.parametrize("name", presets.names())
def test_every_preset_selects_registered_parts(name):
    cfg = presets.resolve(name)
    assert set(cfg) == set(HISTORICAL_DEFAULTS)
    assert cfg["denoiser"] in {b.value for b in BackendId}
    assert cfg["decode"] in decode_presets.PRESETS
    assert cfg["variant"] in (None, "whisper-medasr")
    assert cfg["validate_terms"] in (False, True, "auto")


def test_presets_are_distinct_approaches():
    resolved = [json.dumps(presets.resolve(n), sort_keys=True) for n in presets.names()]
    assert len(set(resolved)) == len(resolved)


def test_a_preset_changes_only_what_it_names():
    assert presets.resolve("regex-only") == {**HISTORICAL_DEFAULTS, "validate_terms": False}
    assert presets.resolve("no-second-pass") == {**HISTORICAL_DEFAULTS, "second_pass": None}
    assert presets.resolve("context-repair") == {**HISTORICAL_DEFAULTS, "validate_terms": True,
                                                 "context_repair": True}


def test_unknown_preset_or_field_is_refused_with_the_valid_names():
    with pytest.raises(ValueError, match="regex-only"):
        presets.resolve("turbo")
    with pytest.raises(ValueError, match="validate_terms"):
        presets.resolve("baseline", {"validate": False})


def test_no_flags_means_no_overrides_and_the_baseline_config():
    args = _parse()
    assert run.typed_overrides(args) == {}
    assert presets.resolve(args.preset, run.typed_overrides(args)) == HISTORICAL_DEFAULTS


def test_typed_flags_override_the_preset_and_nothing_else():
    args = _parse("--preset", "regex-only", "--model", "small-int8", "--decode", "beam5")
    overrides = run.typed_overrides(args)
    assert overrides == {"model": "small-int8", "decode": "beam5"}
    cfg = presets.resolve(args.preset, overrides)
    assert cfg["validate_terms"] is False and cfg["model"] == "small-int8"


def test_flag_translation_matches_the_old_cli_semantics():
    assert run.typed_overrides(_parse("--no-second-pass"))["second_pass"] is None
    # --no-second-pass wins over --second-pass, as it always did.
    assert run.typed_overrides(_parse("--no-second-pass", "--second-pass", "large-v3"))["second_pass"] is None
    assert run.typed_overrides(_parse("--second-pass", "large-v3"))["second_pass"] == "large-v3"
    assert run.typed_overrides(_parse("--llm", "--no-validate", "--context-repair")) == {
        "use_llm": True, "validate_terms": False, "context_repair": True}


def _fake_run_clip(calls):
    def fake(src, arm_dir, denoiser, **kwargs):
        calls.append((denoiser, kwargs))
        return {"transcript": {"stt_engine": "fake"}, "entities": {"drugs": [], "vitals": [], "allergies": []},
                "seconds": {"clean": 0.0, "asr_extract": 0.0}}
    return fake


def _run_main(tmp_path, monkeypatch, *argv):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"")
    calls = []
    monkeypatch.setattr(run, "run_clip", _fake_run_clip(calls))
    out = tmp_path / "out"
    assert run.main([str(clip), "--out", str(out), *argv]) == 0
    return calls, json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def test_default_run_passes_the_historical_arguments_and_records_them(tmp_path, monkeypatch):
    calls, record = _run_main(tmp_path, monkeypatch)
    (denoiser, kwargs), = calls
    assert denoiser == "none"
    assert kwargs == {"model": "medasr", "use_llm": False, "validate_terms": "auto",
                      "second_pass": "auto", "llm_model": None, "context_repair": False,
                      "decode": "greedy"}
    assert record["preset"] == "baseline" and record["config"] == HISTORICAL_DEFAULTS
    assert record["overrides"] == {} and record["arms"] == ["none"]


def test_preset_and_overrides_reach_run_clip_and_the_manifest(tmp_path, monkeypatch):
    calls, record = _run_main(tmp_path, monkeypatch, "--preset", "context-repair",
                              "--model", "small-int8", "--denoiser", "none", "--denoiser", "noisereduce")
    assert [(d, k["model"], k["validate_terms"], k["context_repair"]) for d, k in calls] == [
        ("none", "small-int8", True, True), ("noisereduce", "small-int8", True, True)]
    assert record["preset"] == "context-repair"
    assert record["overrides"] == {"model": "small-int8", "denoiser": ["none", "noisereduce"]}
    assert record["arms"] == ["none", "noisereduce"]


def test_decode_preset_still_refuses_the_medasr_decoder(tmp_path, monkeypatch):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"")
    with pytest.raises(SystemExit):
        run.main([str(clip), "--decode", "beam5", "--out", str(tmp_path / "o")])


def test_manifest_records_the_code_state():
    git = manifest.git_state()
    assert git is not None and len(git["commit"]) == 40 and isinstance(git["dirty"], bool)
    assert "numpy" in manifest.package_versions()
