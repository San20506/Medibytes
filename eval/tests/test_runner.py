from __future__ import annotations

import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from eval.contracts import atomic_write_json, file_sha256, load_json, sha256_bytes
from eval.runner import (
    CATALOG_PATH,
    RunnerError,
    _static_output_snapshot,
    load_downstream_config,
    load_method_spec,
    run_condition,
    run_evaluation,
    validate_method_catalog,
    validate_real_transcript,
)
from eval.metrics import read_pcm16_mono

SAMPLE_RATE = 16_000


def _write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())


def _record(data_root: Path) -> dict[str, object]:
    clean = data_root / "corpus" / "clean.wav"
    noisy = data_root / "corpus" / "noisy.wav"
    time_axis = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    clean_samples = 0.1 * np.sin(2.0 * np.pi * 220.0 * time_axis)
    noisy_samples = clean_samples + 0.005 * np.sin(2.0 * np.pi * 1100.0 * time_axis)
    _write_wav(clean, clean_samples)
    _write_wav(noisy, noisy_samples)
    return {
        "sample_id": "sample-001",
        "group_id": "group-001",
        "bucket": "en-US-proxy",
        "source": {"transcript": "one two three"},
        "clean": {
            "path": "corpus/clean.wav",
            "sha256": file_sha256(clean),
            "sample_count": SAMPLE_RATE,
        },
        "noisy": {
            "path": "corpus/noisy.wav",
            "noise_type": "white",
            "sha256": file_sha256(noisy),
            "sample_count": SAMPLE_RATE,
        },
    }


def _transcript(condition_id: str) -> dict[str, object]:
    return {
        "job_id": condition_id,
        "text": "alpha beta gamma",
        "language": "en",
        "segments": [],
        "normalized_en": "alpha beta gamma",
        "normalizations": [],
        "stt_engine": "faster-whisper:base-cpu-int8",
        "stt_provenance": {
            "provider": "faster-whisper",
            "requested_model": "base-int8",
            "actual_model": "base",
            "device": "cpu",
            "compute_type": "int8",
            "beam_size": 1,
            "runtime_version": "1.2.1",
            "model_hash": None,
            "model_snapshot": None,
            "is_mock": False,
            "job_id": condition_id,
            "word_timestamps": True,
            "temperature": 0.0,
        },
    }


def _entities(condition_id: str = "sample-001-noisy") -> dict[str, object]:
    return {
        "job_id": condition_id,
        "drugs": [],
        "symptoms": [],
        "vitals": [],
        "allergies": [],
        "negations": [],
        "diagnosis": [],
        "followup": [],
    }


def _metadata(
    spec: dict[str, object],
    output: Path,
    none_reference: str,
) -> dict[str, object]:
    digest = file_sha256(output)
    return {
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "input_sample_count": spec["input_samples"],
        "sample_count_before": spec["input_samples"],
        "sample_count_after": spec["input_samples"],
        "input_duration_s": 1.0,
        "output_duration_s": 1.0,
        "vad_ratio": 1.0,
        "output_sample_count": spec["input_samples"],
        "sample_count_drift": 0,
        "duration_drift_s": 0.0,
        "rms_dbfs_before": -20.0,
        "rms_dbfs_after": -19.0,
        "requested_backend": spec["method"],
        "actual_backend": spec["method"],
        "variant_id": spec["variant_id"],
        "model_sha256": None,
        "runtime_version": "test-runtime",
        "source_revision": "b0" if spec["method"] == "none" else "test-revision",
        "native_sample_rate": SAMPLE_RATE,
        "delay_samples": 0,
        "fallback_used": False,
        "input_sha256": file_sha256(Path(str(spec["input_path"]))),
        "output_sha256": digest,
        "processing_time_s": 0.01,
        "preprocess_gain_linear": 1.0,
        "preprocess_peak_scale": 1.0,
        "preprocess_rms_gain_linear": 1.0,
        "preprocess_peak_before": 0.5,
        "preprocess_peak_after": 0.5,
        "output_peak": 0.5,
        "output_clipping_count": 0,
        "enhancement_metadata": {"parameters": dict(spec["enhancement_config"]["parameters"])},
        "preprocess_total_gain_linear": 1.0,
        "preprocess_reference_sha256": "c" * 64,
        "none_reference_sha256": none_reference,
        "provenance": {
            "requested_backend": spec["method"],
            "actual_backend": spec["method"],
            "backend": spec["method"],
            "variant_id": spec["variant_id"],
            "model_sha256": None,
            "runtime_version": "test-runtime",
            "source_revision": "b0" if spec["method"] == "none" else "test-revision",
            "device": "cpu",
            "provider": "test-runtime",
            "parameters": dict(spec["enhancement_config"]["parameters"]),
            "fallback_used": False,
            "config_sha256": str(spec["enhancement_config"]["config_sha256"]),
        },
    }


def _resource(phase: str) -> dict[str, object]:
    return {
        "stage1_wall_s": 0.01,
        "cold_total_s": 0.02,
        "model_load_s": 0.0,
        "inference_s": 0.01,
        "child_cpu_time_s": 0.02,
        "peak_rss_bytes": 1024,
        "peak_vram_bytes": "unavailable",
        "audio_duration_s": 1.0,
        "real_time_factor": 0.01,
    }


def test_catalog_and_downstream_config_are_frozen() -> None:
    catalog = load_json(CATALOG_PATH)
    records = validate_method_catalog(catalog)
    assert len(records) == 12
    assert [record["csv_row"] for record in records] == list(range(2, 14))
    for method in (
        "none",
        "noisereduce",
        "rnnoise",
        "deepfilternet3",
        "dpdfnet2-onnx",
        "sherpa-gtcrn-simple",
        "percepnet",
        "nemo-se-den-sb-16k",
        "speecht5",
        "speex-denoise",
        "speechbrain-metricgan-plus",
        "voicefixer-mode0",
        "cmsis-dsp",
        "spectral-subtraction",
    ):
        spec = load_method_spec(method, Path("/tmp/medibytes-eval-config-test"))
        assert spec["identity"]["variant_id"] == spec["config"]["variant_id"]
        assert spec["identity"]["target_class"] == spec["config"]["target_class"]
        assert spec["identity"]["license"] == spec["config"]["license"]

    downstream, _ = load_downstream_config()
    assert downstream["strict"] is True
    assert downstream["stt"]["model"] == "base-int8"
    assert downstream["cache"] == {
        "output_cache_enabled": False,
        "idempotency_reuse_enabled": False,
    }


def test_strict_transcript_rejects_mock_fallback_and_missing_spans() -> None:
    downstream, _ = load_downstream_config()
    transcript = _transcript("sample-001-noisy")
    entities = _entities()
    validate_real_transcript(transcript, entities, downstream)

    transcript["stt_provenance"]["is_mock"] = True
    with pytest.raises(RunnerError, match="mock"):
        validate_real_transcript(transcript, entities, downstream)
    transcript["stt_provenance"]["is_mock"] = False

    for field, value in (
        ("provider", "mock"),
        ("requested_model", "tiny-int8"),
        ("actual_model", "small"),
        ("device", "cuda"),
        ("compute_type", "float16"),
        ("beam_size", 5),
        ("word_timestamps", False),
        ("temperature", 0.7),
        ("job_id", "other-condition"),
        ("is_mock", True),
    ):
        drifted = _transcript("sample-001-noisy")
        drifted["stt_provenance"][field] = value
        with pytest.raises(RunnerError):
            validate_real_transcript(drifted, _entities(), downstream)

    for field, value in (("word_timestamps", 1), ("temperature", True)):
        drifted = _transcript("sample-001-noisy")
        drifted["stt_provenance"][field] = value
        with pytest.raises(RunnerError):
            validate_real_transcript(drifted, _entities(), downstream)

    drifted = _transcript("sample-001-noisy")
    drifted["job_id"] = 7
    drifted["stt_provenance"]["job_id"] = 7
    with pytest.raises(RunnerError, match="job_id"):
        validate_real_transcript(drifted, _entities(), downstream)

    empty = _transcript("sample-001-noisy")
    empty["text"] = ""
    empty["normalized_en"] = ""
    empty["normalizations"] = []
    validate_real_transcript(empty, _entities(), downstream)

    unnormalized = _transcript("sample-001-noisy")
    unnormalized["normalized_en"] = "different text"
    with pytest.raises(RunnerError, match="frozen normalizer"):
        validate_real_transcript(unnormalized, _entities(), downstream)


    entities["drugs"] = [{"name": "x"}]
    with pytest.raises(RunnerError, match="start_char"):
        validate_real_transcript(transcript, entities, downstream)

    empty_scalars = _entities()
    empty_scalars["diagnosis"] = {}
    empty_scalars["followup"] = {}
    validate_real_transcript(_transcript("sample-001-noisy"), empty_scalars, downstream)


@pytest.mark.parametrize("noise_kind", ["clean", "noisy"])
def test_reference_accuracy_warmup_and_three_repeats_use_fresh_invocations(
    tmp_path: Path,
    noise_kind: str,
) -> None:
    data_root = tmp_path / "data"
    run_root = tmp_path / "run"
    record = _record(data_root)
    method_spec = load_method_spec("noisereduce", data_root)
    downstream, _ = load_downstream_config()
    phases: list[str] = []

    def invoke(spec: dict[str, object], _work_dir: Path, _timeout: float) -> dict[str, object]:
        phases.append(str(spec["phase"]))
        input_path = Path(str(spec["input_path"]))
        output_path = Path(str(spec["output_path"]))
        shutil.copyfile(input_path, output_path)
        reference_path = _work_dir / "input-reference.wav"
        none_reference = file_sha256(output_path if spec["method"] == "none" else reference_path)
        if spec["method"] != "none":
            with wave.open(str(output_path), "rb") as source:
                samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").copy()
            samples[100] = np.int16(min(32767, int(samples[100]) + 64))
            _write_wav(output_path, samples.astype(np.float64) / 32768.0)
        result: dict[str, object] = {
            "status": "ok",
            "phase": spec["phase"],
            "enhanced_path": str(output_path),
            "enhancement": _metadata(spec, output_path, none_reference),
            "resources": _resource(str(spec["phase"])),
            "downstream_status": "not_run",
        }
        if spec.get("downstream"):
            result["transcript"] = _transcript(str(spec["condition_id"]))
            result["entities"] = _entities(str(spec["condition_id"]))
            result["downstream_status"] = "ok"
        return result

    before = _static_output_snapshot()
    status = run_condition(
        record,
        noise_kind,
        run_root=run_root,
        data_root=data_root,
        method_spec=method_spec,
        downstream=downstream,
        invoke_worker=invoke,
    )

    assert phases == [
        "clean-reference",
        "input-reference",
        "accuracy",
        "warmup",
        "repeat-1",
        "repeat-2",
        "repeat-3",
    ]
    assert status["status"] == "ok"
    assert len(status["repeat_hashes"]) == 4
    assert len(set(status["repeat_hashes"])) == 1
    assert status["actual_backend"] == "noisereduce"
    assert status["condition_kind"] == noise_kind
    assert status["noise_type"] == ("clean" if noise_kind == "clean" else "white")
    assert _static_output_snapshot() == before
    condition_dir = (
        run_root
        / "arms"
        / "noisereduce"
        / "3.0.2-control"
        / "conditions"
        / f"sample-001-{noise_kind}"
    )
    assert {path.name for path in condition_dir.iterdir()} == {
        "enhanced.wav",
        "enhancement.json",
        "transcript.json",
        "entities.json",
        "fields.json",
        "resources.json",
        "metrics.json",
        "status.json",
    }
    metrics = load_json(condition_dir / "metrics.json")
    assert len(metrics["audio"]["aligned_reference_sha256"]) == 64
    assert metrics["audio"]["reference_input_sha256"] == record["clean"]["sha256"]
    if noise_kind == "clean":
        assert metrics["audio"]["candidate_none_reference_sha256"] == metrics["audio"]["none_reference_sha256"]
    else:
        assert metrics["audio"]["candidate_none_reference_sha256"] != metrics["audio"]["none_reference_sha256"]

    stale_status = load_json(condition_dir / "status.json")
    stale_status["group_id"] = "stale-group"
    atomic_write_json(condition_dir / "status.json", stale_status)
    with pytest.raises(RunnerError, match="does not match current condition"):
        run_condition(
            record,
            noise_kind,
            run_root=run_root,
            data_root=data_root,
            method_spec=method_spec,
            downstream=downstream,
            invoke_worker=invoke,
        )


def test_metrics_reuse_canonical_none_reference_without_second_gain(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    run_root = tmp_path / "run"
    record = _record(data_root)
    method_spec = load_method_spec("noisereduce", data_root)
    downstream, _ = load_downstream_config()

    def invoke(spec: dict[str, object], work_dir: Path, _timeout: float) -> dict[str, object]:
        with wave.open(str(spec["input_path"]), "rb") as source:
            raw = np.frombuffer(
                source.readframes(source.getnframes()), dtype="<i2"
            ).astype(np.float64) / 32768.0
        gain = 1.0 if spec["phase"] == "clean-reference" else 2.0
        output_path = Path(str(spec["output_path"]))
        _write_wav(output_path, raw * gain)
        none_reference = file_sha256(
            output_path if spec["method"] == "none" else work_dir / "input-reference.wav"
        )
        enhancement = _metadata(spec, output_path, none_reference)
        for key in (
            "preprocess_gain_linear",
            "preprocess_rms_gain_linear",
            "preprocess_total_gain_linear",
        ):
            enhancement[key] = gain
        result: dict[str, object] = {
            "status": "ok",
            "phase": spec["phase"],
            "enhanced_path": str(output_path),
            "enhancement": enhancement,
            "resources": _resource(str(spec["phase"])),
            "downstream_status": "not_run",
        }
        if spec.get("downstream"):
            result["transcript"] = _transcript(str(spec["condition_id"]))
            result["entities"] = _entities(str(spec["condition_id"]))
            result["downstream_status"] = "ok"
        return result

    status = run_condition(
        record,
        "noisy",
        run_root=run_root,
        data_root=data_root,
        method_spec=method_spec,
        downstream=downstream,
        invoke_worker=invoke,
    )

    assert status["status"] == "ok"
    condition_dir = (
        run_root
        / "arms"
        / "noisereduce"
        / "3.0.2-control"
        / "conditions"
        / "sample-001-noisy"
    )
    metrics = load_json(condition_dir / "metrics.json")
    canonical_clean = read_pcm16_mono(data_root / "corpus" / "clean.wav")
    expected_hash = sha256_bytes(np.asarray(canonical_clean, dtype="<f4").tobytes())
    assert metrics["audio"]["aligned_reference_sha256"] == expected_hash
    assert metrics["audio"]["reference_preprocess_total_gain_linear"] == 1.0
    assert metrics["audio"]["input_preprocess_total_gain_linear"] == 2.0


def test_oom_is_retained_as_terminal_without_fake_audio(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    run_root = tmp_path / "run"
    record = _record(data_root)
    method_spec = load_method_spec("none", data_root)
    downstream, _ = load_downstream_config()
    calls = 0

    def invoke(_spec: dict[str, object], _work_dir: Path, _timeout: float) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "status": "oom",
            "phase": "reference",
            "error": {
                "type": "MemoryError",
                "message": "worker exhausted memory",
                "traceback": "",
            },
        }

    status = run_condition(
        record,
        "clean",
        run_root=run_root,
        data_root=data_root,
        method_spec=method_spec,
        downstream=downstream,
        invoke_worker=invoke,
    )
    assert calls == 1
    assert status["status"] == "oom"
    assert status["enhanced"] is None
    condition_dir = run_root / "arms" / "none" / "identity-v1" / "conditions" / "sample-001-clean"
    assert not (condition_dir / "enhanced.wav").exists()
    assert (condition_dir / "metrics.json").is_file()
    assert (condition_dir / "status.json").is_file()


def test_run_evaluation_rejects_symlinked_results_root(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    runs_root = data_root / "runs"
    runs_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_root = runs_root / "linked-run"
    linked_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RunnerError, match="symlink-free"):
        run_evaluation(
            method="none",
            run_id="linked-run",
            data_root_value=data_root,
            results_root=linked_root,
        )


def test_blocked_run_evaluation_assigns_and_verifies_run_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    results_root = data_root / "runs" / "blocked-run"
    monkeypatch.setattr("eval.runner._load_and_verify_corpus", lambda _manifest, _root: [])

    result = run_evaluation(
        method="percepnet",
        run_id="blocked-run",
        data_root_value=data_root,
        results_root=results_root,
    )

    assert result["status"] == "complete"
    assert result["counts"] == {
        "expected_conditions": 0,
        "recorded_conditions": 0,
        "ok": 0,
        "failed": 0,
    }
    assert (results_root / "run.json").is_file()
    assert (results_root / "arms" / "percepnet" / "unofficial-8ffae433" / "report.json").is_file()
