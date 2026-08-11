import app.tools as tools


class FakeResponse:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._data


def test_predict_builds_payload_and_sets_kserve_relation_attribute(span_exporter, monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse({"predictions": [0]})

    monkeypatch.setattr(tools.httpx, "post", fake_post)

    result = tools.predict(5.1, 3.5, 1.4, 0.2, tool_call_id="tc1")

    assert "0" in result
    assert captured["json"] == {"instances": [[5.1, 3.5, 1.4, 0.2]]}
    assert captured["url"] == tools.KSERVE_PREDICT_URL

    spans = span_exporter.get_finished_spans()
    client = [s for s in spans if s.name == "predict sklearn-iris"]
    assert client, "expected a 'predict sklearn-iris' client span"
    assert client[0].attributes["kserve.inference.service"] == "sklearn-iris"


def test_list_models_calls_registry_and_returns_names(span_exporter, monkeypatch):
    captured = {}

    def fake_get(url, timeout=None):
        captured["url"] = url
        return FakeResponse({"items": [{"name": "iris"}, {"name": "mnist"}]})

    monkeypatch.setattr(tools.httpx, "get", fake_get)

    result = tools.list_models(tool_call_id="tc2")

    assert "iris" in result and "mnist" in result
    assert captured["url"].endswith("/api/model_registry/v1alpha3/registered_models")


def test_descriptions_present():
    assert "predict" in tools.TOOL_DESCRIPTIONS
    assert "list_models" in tools.TOOL_DESCRIPTIONS
