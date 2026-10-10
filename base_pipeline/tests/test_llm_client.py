"""llm_client against a real HTTP server speaking the OpenAI-compatible API.

A stdlib server on an ephemeral port stands in for Ollama / llama-server, so the
wire format is exercised for real rather than mocked at the function boundary.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from base_pipeline import llm_client, term_validate


class _Server:
    def __init__(self, models, reply):
        self.models, self.reply, self.requests = models, reply, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def _send(self, code, body):
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                self._send(200, {"data": [{"id": m} for m in outer.models]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body, self.headers.get("Authorization")))
                self._send(200, {"choices": [{"message": {"content": outer.reply}}]})

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(monkeypatch):
    made = []

    def make(models=("llama3.2:3b",), reply='{"ok": true}'):
        srv = _Server(models, reply)
        made.append(srv)
        monkeypatch.setenv("MEDIBYTES_LLM_URL", srv.url)
        return srv

    yield make
    for srv in made:
        srv.close()


def test_available_when_the_model_is_served(server):
    server()
    assert llm_client.llm_available("llama3.2:3b") == (True, "")


def test_unserved_model_is_reported_not_silently_substituted(server):
    server(models=("a", "b"))
    ok, why = llm_client.llm_available("llama3.2:3b")
    assert not ok and "not served" in why


def test_a_lone_served_model_is_accepted_under_any_name(server):
    # llama-server names the model after the GGUF file, not what the caller asks for.
    server(models=("some-file.gguf",))
    assert llm_client.llm_available("llama3.2:3b")[0]


def test_dead_server_is_a_reason_not_an_exception(monkeypatch):
    monkeypatch.setenv("MEDIBYTES_LLM_URL", "http://127.0.0.1:9/v1")
    ok, why = llm_client.llm_available()
    assert not ok and "no LLM server" in why


def test_null_or_malformed_model_listing_is_a_reason_not_an_exception(monkeypatch):
    # Observed: a local shim answers 200 with {"object": "list", "data": null},
    # which used to raise TypeError and fail the whole CLI run per clip.
    real = llm_client._request
    for bad in ({"object": "list", "data": None}, {"object": "list"}, [1, 2], None):
        llm_client._request = lambda *a, **k: bad  # noqa: E731
        try:
            ok, why = llm_client.llm_available()
        finally:
            llm_client._request = real
        assert ok is False and isinstance(why, str) and why


def test_complete_json_sends_a_deterministic_json_request(server, monkeypatch):
    srv = server(reply='{"x": 1}')
    monkeypatch.setenv("MEDIBYTES_LLM_KEY", "secret")
    assert llm_client.complete_json("hi", "m", max_tokens=7) == '{"x": 1}'
    path, body, auth = srv.requests[0]
    assert path == "/v1/chat/completions" and auth == "Bearer secret"
    assert body["model"] == "m" and body["temperature"] == 0 and body["max_tokens"] == 7
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_complete_json_on_a_dead_server_raises_valueerror(monkeypatch):
    monkeypatch.setenv("MEDIBYTES_LLM_URL", "http://127.0.0.1:9/v1")
    with pytest.raises(ValueError, match="no LLM server"):
        llm_client.complete_json("hi")


def test_the_validator_runner_probes_availability_once_not_per_row(server):
    srv = server(reply='{"v": 1}')
    run = term_validate._llm_runner("llama3.2:3b", timeout=5)
    gets = []
    real = llm_client._request

    def counting(path, payload=None, timeout=15):
        if payload is None:
            gets.append(path)
        return real(path, payload, timeout)

    llm_client._request = counting
    try:
        assert run("a") == {"v": 1} and run("b") == {"v": 1} and run("c") == {"v": 1}
    finally:
        llm_client._request = real
    assert gets == ["/models"] and len(srv.requests) == 3


def test_a_slow_server_is_reported_as_a_timeout_not_as_missing(monkeypatch):
    import socket
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)  # accepts the connection, never answers
    try:
        monkeypatch.setenv("MEDIBYTES_LLM_URL", f"http://127.0.0.1:{sock.getsockname()[1]}/v1")
        with pytest.raises(llm_client.LLMTimeout, match="timed out after 1"):
            llm_client.complete_json("{}", timeout=1)
    finally:
        sock.close()


def test_timeout_budget_is_overridable_and_ignores_garbage(monkeypatch):
    monkeypatch.delenv("MEDIBYTES_LLM_TIMEOUT", raising=False)
    assert llm_client.timeout_for(300) == 300
    monkeypatch.setenv("MEDIBYTES_LLM_TIMEOUT", "600")
    assert llm_client.timeout_for(300) == 600.0
    monkeypatch.setenv("MEDIBYTES_LLM_TIMEOUT", "soon")
    assert llm_client.timeout_for(300) == 300
