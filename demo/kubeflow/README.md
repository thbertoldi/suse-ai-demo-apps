# Kubeflow / KServe Component-Relations Demo

Demonstrates the 2.2.3 component relations: one model (iris) related from both
the app layer and the pipeline layer.

## Prerequisites
- Kubeflow + KServe installed (see `knowledge/KUBEFLOW_INSTALL_RUNBOOK.md` in the
  observability-extension repo).
- Stackpack 2.2.3 uploaded; branch collector (`ghcr.io/suse/suse-ai-opentelemetry-collector:0.156.0`) running.
- The `sklearn-iris` InferenceService already running in `kserve-test`.
- Demo apps deployed via `helm/suse-ai-demo` in the `suse-private-ai` namespace.

## 0. Verify the model-registry REST path (do first)
```bash
kubectl -n kubeflow port-forward svc/model-registry-service 8080:8080 &
curl -s localhost:8080/api/model_registry/v1alpha3/registered_models | head
# If this 404s, adjust MODEL_REGISTRY_URL / the path in steps.py + tools.py.
```

## 1. Always-on app edges
```bash
# After CI builds the agent-service image, roll it out to pick up the new tools:
kubectl -n suse-private-ai set env deploy/agent-service \
  KSERVE_PREDICT_URL=http://sklearn-iris.kserve-test.svc.cluster.local/v1/models/sklearn-iris:predict \
  MODEL_REGISTRY_URL=http://model-registry-service.kubeflow.svc.cluster.local:8080
kubectl -n suse-private-ai rollout restart deploy/agent-service deploy/traffic-gen
```
Wait a few minutes for traffic, then in SUSE Observability topology confirm:
- `agent-service ──▶ inference-engine.kserve`
- `agent-service ──▶ ml-registry.kubeflow`

## 2. Pipeline edges (re-runnable one-shot)
```bash
cd demo/kubeflow
docker build -t ghcr.io/thbertoldi/suse-ai-demo-iris-pipeline:latest .
docker push ghcr.io/thbertoldi/suse-ai-demo-iris-pipeline:latest
python -m pipeline            # produces iris_pipeline.yaml
# Upload iris_pipeline.yaml via the Kubeflow Pipelines UI and create a run,
# OR submit with the KFP SDK against the in-cluster endpoint.
```
After the run completes, confirm:
- `kubeflow-pipelines ──▶ ml-registry.kubeflow`
- `kubeflow-pipelines ──▶ inference-engine.kserve`

## 3. Full picture
Filter topology to the new components; all four edges converge on the iris model.
Click `inference-engine.kserve` → `request_predict_seconds_count{model_name="sklearn-iris"}`
climbs with agent traffic. `ml-registry.kubeflow` renders (dark on a bare install).

## Troubleshooting
- **No agent→kserve edge:** confirm the `predict sklearn-iris` span reaches the
  collector and `transform/kubeflow-relations` is in the `traces/kubeflow-relations`
  pipeline. Check the agent can reach the predictor across namespaces.
- **No agent→registry edge:** the httpx GET must actually hit a URL containing
  `model-registry`; confirm `MODEL_REGISTRY_URL` and that the REST path returns 200.
- **No KFP edges:** confirm the step pods carry `OTEL_RESOURCE_ATTRIBUTES` with
  `suse.ai.component.name=kubeflow-pipelines` and that their spans reach the collector.
- **Kubeflow OIDC on a full install:** the istio ingress gateway enforces OIDC, so
  in-cluster calls to the KServe route and to `model-registry-service` get bounced
  (302 → `/dex/auth`, or 403). The topology edges still form — they are built from
  the OTel spans (the `predict` CLIENT span's `kserve.inference.service` attribute
  and the httpx span's `model-registry` URL), not from HTTP success. To get real
  `200` predictions in-cluster, target the Knative revision service directly, e.g.
  `http://sklearn-iris-predictor-<revision>.kserve-test.svc.cluster.local/...`
  (`kubectl -n kserve-test get svc`), which bypasses the gateway.
