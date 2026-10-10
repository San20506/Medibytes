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
