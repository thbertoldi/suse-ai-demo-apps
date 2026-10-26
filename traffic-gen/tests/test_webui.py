"""Exercise authentication and both Open WebUI response lifecycles over HTTP."""

import http.server
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest


spec = importlib.util.spec_from_file_location("webui_traffic", Path(__file__).parents[1] / "webui.py")
webui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(webui)


class TrafficTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        requests = self.requests

        class Handler(http.server.BaseHTTPRequestHandler):
            models_loaded = False

            def log_message(self, *args):
                pass

            def do_GET(self):
                requests.append((self.path, None, self.headers.get("Authorization")))
                Handler.models_loaded = True
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": "llama3.2"}, {"id": "empty"}]}).encode())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, body, self.headers.get("Authorization")))
                self.send_response(200)
                if self.path.endswith("/signin"):
                    result = {"role": "admin" if body.get("email") == "admin@example.com" else "user",
                              "token": "ordinary-demo-user"}
                elif self.path.endswith("/completed"):
                    result = body
                elif not Handler.models_loaded or body["model"] == "empty":
                    result = None
                elif body["stream"]:
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for chunk in ({"choices": [{"delta": {"content": "A "}}]},
                                  {"choices": [{"delta": {"content": "pod."}}]},
                                  {"usage": {"prompt_tokens": 3, "completion_tokens": 2}}):
                        self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                else:
                    result = {"choices": [{"message": {"content": "A pod."}}],
                              "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.directory = tempfile.TemporaryDirectory()
        self.credentials = Path(self.directory.name) / "credentials.json"
        self.credentials.write_text('{"email":"demo@example.com","password":"fixture"}')

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def test_stream_and_nonstream_each_finish_once_with_usage_and_session_id(self):
        client = webui.Client(f"http://127.0.0.1:{self.server.server_port}", self.credentials)
        for stream in (False, True):
            result = client.chat("llama3.2", "What is a pod?", stream)
            self.assertEqual(result["usage"], {"prompt_tokens": 3, "completion_tokens": 2})
            self.assertEqual(result["stream"], stream)
        self.assertEqual([r[0] for r in self.requests].count("/api/v1/auths/signin"), 1)
        self.assertEqual([r[0] for r in self.requests].count("/api/models"), 2)
        for index, (path, _, _) in enumerate(self.requests):
            if path == "/api/chat/completions":
                self.assertEqual(self.requests[index - 1][0], "/api/models")
        completed = [r for r in self.requests if r[0] == "/api/chat/completed"]
        self.assertEqual(len(completed), 2)
        self.assertNotEqual(completed[0][1]["id"], completed[1][1]["id"])
        for _, body, auth in completed:
            self.assertTrue(body["chat_id"].startswith("local:"))
            self.assertEqual(body["session_id"], "")
            self.assertEqual(body["messages"][-1]["content"], "A pod.")
            self.assertEqual(auth, "Bearer ordinary-demo-user")

    def test_administrator_credentials_are_rejected(self):
        self.credentials.write_text('{"email":"admin@example.com","password":"fixture"}')
        client = webui.Client(f"http://127.0.0.1:{self.server.server_port}", self.credentials)
        with self.assertRaisesRegex(RuntimeError, "ordinary user"):
            client.chat("llama3.2", "A test")
        self.assertEqual([r[0] for r in self.requests], ["/api/v1/auths/signin"])

    def test_null_provider_response_does_not_send_a_success_callback(self):
        client = webui.Client(f"http://127.0.0.1:{self.server.server_port}", self.credentials)
        with self.assertRaisesRegex(RuntimeError, "no completion"):
            client.chat("empty", "A test")
        self.assertNotIn("/api/chat/completed", [r[0] for r in self.requests])

    def test_unavailable_model_does_not_attempt_generation(self):
        client = webui.Client(f"http://127.0.0.1:{self.server.server_port}", self.credentials)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            client.chat("not-shared-with-demo-user", "A test")
        self.assertNotIn("/api/chat/completions", [r[0] for r in self.requests])


if __name__ == "__main__":
    unittest.main()
