"""The evaluation trigger rides on the pipeline server and fails soft when the corpus is absent."""
import pytest

pytest.importorskip("fastapi.testclient")


def test_eval_route_is_added_to_the_pipeline_app_and_reports_an_unstaged_corpus(tmp_path):
    from fastapi.testclient import TestClient

    from clinical_eval.server import app

    paths = {route.path for route in app.routes}
    assert {"/api/eval/run", "/api/audio/jobs", "/api/health"} <= paths

    response = TestClient(app).post(
        "/api/eval/run", data={"data_root": str(tmp_path), "denoisers": "none", "decoders": "mock"})
    assert response.status_code == 503
    assert "corpus not staged" in response.json()["detail"]
