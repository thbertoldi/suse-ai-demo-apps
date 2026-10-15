"""Bounded chat traffic through Open WebUI 0.6.41, using a dedicated user."""

import json
import logging
import os
from pathlib import Path
import signal
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid


LOG = logging.getLogger("demo.webui")
STOP = threading.Event()


class Client:
    def __init__(self, base_url, credentials_file, timeout=120):
        self.base_url = base_url.rstrip("/")
        self.credentials_file = Path(credentials_file)
        self.timeout = timeout
        self.token = None

    def request(self, path, body=None, authenticate=True):
        headers = {"Content-Type": "application/json"}
        if authenticate:
            if not self.token:
                self.login()
            headers["Authorization"] = "Bearer " + self.token
        data = json.dumps(body).encode() if body is not None else None
        return urlopen(Request(self.base_url + path, data, headers), timeout=self.timeout)

    def login(self):
        credentials = json.loads(self.credentials_file.read_text())
        with self.request("/api/v1/auths/signin", credentials, authenticate=False) as response:
            session = json.load(response)
        if session.get("role") != "user":
            raise RuntimeError("Traffic requires a dedicated ordinary user account")
        self.token = session["token"]

    def chat(self, model, prompt, stream=False, max_tokens=64):
        # Like the browser, load the accessible models before selecting one.
        # On 0.6.41 a worker restart can leave the Ollama routing cache empty
        # even when persisted custom models already exist in the main registry.
        with self.request("/api/models") as response:
            available = json.load(response)
        if model not in {item["id"] for item in available.get("data", [])}:
            raise RuntimeError("The configured demo model is unavailable to this user")
        # Open WebUI's supported temporary-chat form avoids accumulating saved
        # conversations and passes ordinary-user ownership checks on 0.6.41.
        chat_id, message_id = "local:" + str(uuid.uuid4()), str(uuid.uuid4())
        user = {"role": "user", "content": prompt}
        body = {"model": model, "messages": [user], "stream": stream,
                "chat_id": chat_id, "id": message_id,
                "params": {"temperature": 0, "num_predict": max_tokens}}
        text, usage, response_model = "", {}, model
        with self.request("/api/chat/completions", body) as response:
            if stream:
                for raw in response:
                    if not raw.startswith(b"data:"):
                        continue
                    data = raw[5:].strip()
                    if data == b"[DONE]":
                        break
                    value = json.loads(data)
                    if value.get("error"):
                        raise RuntimeError("Open WebUI returned a stream error")
                    for choice in value.get("choices", []):
                        text += choice.get("delta", {}).get("content") or ""
                    if value.get("usage"):
                        usage = value["usage"]
                    response_model = value.get("model", response_model)
            else:
                value = json.load(response)
                if not isinstance(value, dict) or value.get("error") or not value.get("choices"):
                    raise RuntimeError("Open WebUI returned no completion")
                text = value["choices"][0]["message"]["content"]
                usage = value.get("usage", {})
                response_model = value.get("model", model)
        assistant = {"role": "assistant", "content": text, "id": message_id, "usage": usage}
        # 0.6.41 requires this lifecycle callback and its session_id field.
        with self.request("/api/chat/completed", {
            "model": model, "chat_id": chat_id, "id": message_id,
            "session_id": "", "messages": [user, assistant],
        }) as response:
            json.load(response)
        if not text.strip():
            raise RuntimeError("Open WebUI returned an empty answer")
        return {"chat_id": chat_id, "message_id": message_id, "model": response_model,
                "stream": stream, "usage": usage}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    interval = max(1, int(os.environ.get("WEBUI_INTERVAL_SECONDS", "60")))
    retries = max(0, min(3, int(os.environ.get("WEBUI_MAX_RETRIES", "1"))))
    stream_every = max(1, int(os.environ.get("WEBUI_STREAM_EVERY", "3")))
    max_tokens = max(1, min(512, int(os.environ.get("WEBUI_MAX_TOKENS", "64"))))
    model = os.environ.get("WEBUI_MODEL", "llama3.2:latest")
    client = Client(os.environ["WEBUI_BASE_URL"], os.environ["WEBUI_CREDENTIALS_FILE"])
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: STOP.set())
    sequence = 0
    while not STOP.is_set():
        sequence += 1
        started = time.monotonic()
        for attempt in range(1, retries + 2):
            if STOP.is_set():
                break
            try:
                result = client.chat(model, "Explain one benefit of Kubernetes in one short sentence.",
                                     stream=sequence % stream_every == 0, max_tokens=max_tokens)
                LOG.info(json.dumps({"sequence": sequence, "attempt": attempt, "outcome": "success",
                                     "elapsed_seconds": round(time.monotonic() - started, 3), **result}))
                break
            except Exception as error:
                # Each retry is a distinct provider attempt, reported separately.
                # No passwords, tokens, prompts or exception response bodies logged.
                if isinstance(error, HTTPError) and error.code == 401:
                    client.token = None
                LOG.error(json.dumps({"sequence": sequence, "attempt": attempt, "outcome": "error",
                                      "error_type": type(error).__name__, "http_status": getattr(error, "code", None)}))
                if attempt <= retries:
                    STOP.wait(min(5 * attempt, interval))
        STOP.wait(max(0, interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
