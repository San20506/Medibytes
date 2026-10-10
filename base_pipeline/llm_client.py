"""One client for whichever local LLM server is running.

The pipeline used to spawn `ollama run` for every call, paying process start
and a model load each time.  This talks to a persistent server over the
OpenAI-compatible HTTP API instead, which Ollama, llama.cpp's `llama-server`,
vLLM and LM Studio all expose - so the model stays resident and "any local LLM"
is a URL, not a code change.

    MEDIBYTES_LLM_URL    base URL, default http://127.0.0.1:11434/v1 (Ollama)
    MEDIBYTES_LLM_MODEL  model name, default llama3.2:3b
    MEDIBYTES_LLM_KEY    bearer token, only for servers that demand one
    OPENAI_API_KEY / CHATGPT_API_KEY
                         used as the token when the URL is api.openai.com
                         (never sent to any other host)
    MEDIBYTES_LLM_TIMEOUT  seconds for the long discharge call (default 300;
                         a 3B model on CPU takes ~130 s for one note)

    OpenAI:     MEDIBYTES_LLM_URL=https://api.openai.com/v1 MEDIBYTES_LLM_MODEL=gpt-4.1
                (or run.py --llm-env FILE --llm-url ... --llm-model ...)
                Non-reasoning models only: gpt-5/o-series reject temperature
                and max_tokens.
    llama.cpp:  llama-server -m model.gguf --port 8080
                MEDIBYTES_LLM_URL=http://127.0.0.1:8080/v1
    Ollama:     OLLAMA_KEEP_ALIVE=1h ollama serve   (keeps the model loaded)

Every failure raises ValueError with a short reason.  Callers already treat
that as "no LLM, fall back to regex", so nothing here may be fatal.
"""
import json
import os
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "llama3.2:3b"


class LLMTimeout(ValueError):
    """The server answered the probe but did not finish in time.

    A ValueError, so every caller's "no LLM, fall back" path still holds.
    """


def timeout_for(default):
    """Per-call budget, overridable with MEDIBYTES_LLM_TIMEOUT (seconds)."""
    try:
        return max(1.0, float(os.environ["MEDIBYTES_LLM_TIMEOUT"]))
    except (KeyError, ValueError):
        return default


def base_url():
    return os.environ.get("MEDIBYTES_LLM_URL", DEFAULT_URL).rstrip("/")


def default_model():
    return os.environ.get("MEDIBYTES_LLM_MODEL", DEFAULT_MODEL)


def _request(path, payload=None, timeout=15):
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("MEDIBYTES_LLM_KEY")
    if not key and base_url().startswith("https://api.openai.com/"):
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CHATGPT_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base_url() + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="ignore"))
    except urllib.error.HTTPError as e:
        raise ValueError(f"LLM server HTTP {e.code}") from None
    except TimeoutError:
        # A live server that is slow is not a dead one: say which, so a CPU
        # box that needs a longer budget is not sent hunting for a server.
        raise LLMTimeout(f"LLM server at {base_url()} timed out after {timeout}s") from None
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            raise LLMTimeout(f"LLM server at {base_url()} timed out after {timeout}s") from None
        raise ValueError(f"no LLM server at {base_url()}") from None
    except OSError:
        raise ValueError(f"no LLM server at {base_url()}") from None
    except json.JSONDecodeError:
        raise ValueError("LLM server returned non-JSON") from None


def llm_available(model=None):
    """(ok, reason).  ok means the server answers and serves `model`."""
    model = model or default_model()
    try:
        listing = _request("/models", timeout=3)
    except ValueError as e:
        return False, str(e)
    if not isinstance(listing, dict):
        return False, "LLM server returned non-JSON"
    # A server may answer 200 with `"data": null` (observed on a local
    # OpenAI-compatible shim) - that is "no models listed", not a crash.
    names = [m.get("id", "") for m in listing.get("data") or [] if isinstance(m, dict)]
    # llama-server reports whatever file it loaded under an arbitrary id, so a
    # single served model is accepted as the one asked for.
    if model in names or len(names) == 1:
        return True, ""
    return False, f"model {model} not served"


def complete_json(prompt, model=None, timeout=60, max_tokens=1024):
    """Run one prompt, return the model's reply text (expected to be JSON)."""
    reply = _request("/chat/completions", {
        "model": model or default_model(),
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }, timeout=timeout)
    try:
        return reply["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise ValueError("LLM reply had no message") from None
