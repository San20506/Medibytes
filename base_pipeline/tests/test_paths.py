from pathlib import Path

from base_pipeline import paths


def test_data_root_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV, str(tmp_path))
    assert paths.data_root() == tmp_path


def test_data_root_default_is_under_the_home_directory(monkeypatch):
    monkeypatch.delenv(paths.ENV, raising=False)
    assert paths.data_root() == Path.home() / ".local" / "share" / "medibytes-eval"
    monkeypatch.setenv(paths.ENV, "  ")
    assert paths.data_root() == Path.home() / ".local" / "share" / "medibytes-eval"


def test_data_root_expands_a_leading_tilde(monkeypatch):
    monkeypatch.setenv(paths.ENV, "~/elsewhere")
    assert paths.data_root() == Path.home() / "elsewhere"


def test_shipped_denoiser_configs_hold_data_root_relative_paths():
    import json

    configs = sorted((Path(paths.__file__).parent / "denoise_backends").glob("*.config.json"))
    assert configs, "expected shipped backend configs"
    for config in configs:
        model_path = json.loads(config.read_text(encoding="utf-8")).get("model_path")
        assert model_path is None or not Path(model_path).is_absolute(), config.name


def test_denoiser_config_resolves_the_shipped_model_path_against_the_data_root(monkeypatch, tmp_path):
    from base_pipeline.denoise import BackendId
    from base_pipeline.run import denoiser_config

    monkeypatch.setenv(paths.ENV, str(tmp_path))
    config = denoiser_config(BackendId.SHERPA_GTCRN_SIMPLE)
    assert config.model_path == tmp_path / "models" / "sherpa-onnx" / "gtcrn_simple.onnx"
