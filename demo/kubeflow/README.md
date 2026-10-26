# Kubeflow model-lifecycle demo

This demo runs a real Iris model lifecycle through Kubeflow Pipelines, Kubeflow
Model Registry, and KServe while emitting OpenTelemetry traces and metrics for
the SUSE AI Observability StackPack.

## Lifecycle

Each uncached pipeline run:

1. Writes the Iris dataset as a KFP `Dataset` artifact.
2. Trains a scikit-learn model and publishes `Model`, `Metrics`, and
   `ClassificationMetrics` artifacts.
3. Enforces a configurable accuracy gate.
4. Creates or reuses the `iris` RegisteredModel, then creates the run-specific
   ModelVersion and ModelArtifact in Kubeflow Model Registry.
5. Deploys the exact KFP model artifact to a KServe `InferenceService`. KFP's
   `minio://` artifact URI is normalized to `s3://` for KServe's storage
   initializer, without copying the model.
6. Waits until KServe's latest-created revision is also its latest-ready
   revision, binds the stable `suse-ai-sklearn-iris` Service to that exact
   revision, and executes a real prediction smoke test.

The revision-pinned Service bypasses the browser-facing Kubeflow OIDC route and
prevents the agent from temporarily reaching an older model during a rollout.

## Prerequisites

- Kubeflow Pipelines, Model Registry, and KServe are installed.
- The profile namespace exists (default: `kubeflow-user-example-com`).
- The demo apps Helm chart is deployed in `suse-private-ai`.
- StackPack 2.2.8 or newer and the SUSE AI collector configuration are active.
- The profile service account can manage namespaced InferenceServices, Secrets,
  ServiceAccounts, and Services. The chart provisions this access through
  `kubeflowPipeline.rbac`.
- The `suse-ai-registry` pull secret exists in the profile namespace for the
  private SUSE KServe runtime image.

Deploy or update the chart before running the pipeline:

```bash
helm upgrade --install suse-ai-demo ./helm/suse-ai-demo \
  --namespace suse-private-ai --create-namespace \
  --set kubeflowPipeline.rbac.enabled=true \
  --set kubeflowPipeline.rbac.serviceAccountNamespace=kubeflow-user-example-com \
  --set kubeflowPipeline.rbac.serviceAccountName=default-editor \
  --set kubeflowPipeline.rbac.kserveNamespace=kubeflow-user-example-com
```

The pipeline, rather than Helm, owns `suse-ai-sklearn-iris` because its selector
must be updated atomically to the revision created by each run.

## Verify Model Registry access

The SUSE Kubeflow demo AuthorizationPolicy requires an `Authorization` header.
The default token is `demo`; use a Secret-backed value outside a demo cluster.

```bash
kubectl -n kubeflow port-forward svc/model-registry-service 18080:8080

curl -fsS -H 'Authorization: Bearer demo' \
  http://127.0.0.1:18080/api/model_registry/v1alpha3/registered_models
```

## Build and submit

Use an immutable image tag so every uploaded pipeline version is reproducible:

```bash
cd demo/kubeflow
python -m venv .venv
.venv/bin/pip install -r requirements.txt

export TAG=demo-$(date -u +%Y%m%d-%H%M%S)
echo "${TAG}"
docker buildx build --platform linux/amd64 \
  -t ghcr.io/thbertoldi/suse-ai-demo-iris-pipeline:${TAG} \
  --push .

kubectl -n kubeflow port-forward svc/ml-pipeline 18889:8888
```

In another terminal:

```bash
cd demo/kubeflow
export TAG=demo-YYYYMMDD-HHMMSS  # same value printed in the build terminal
IRIS_PIPELINE_IMAGE=ghcr.io/thbertoldi/suse-ai-demo-iris-pipeline:${TAG} \
KFP_HOST=http://127.0.0.1:18889 \
  .venv/bin/python submit.py
```

`submit.py` compiles the pipeline, uploads a new version, disables cache, submits
the run to the selected profile, waits for completion, and exits non-zero on
failure. Its defaults target `kubeflow-user-example-com` and `default-editor`.
Override `KFP_NAMESPACE`, `KFP_USER_ID`, or `KFP_SERVICE_ACCOUNT` when needed.

To keep lifecycle telemetry fresh, add `--recurring`. The command creates the
named recurring run only if it does not already exist:

```bash
IRIS_PIPELINE_IMAGE=ghcr.io/thbertoldi/suse-ai-demo-iris-pipeline:${TAG} \
KFP_HOST=http://127.0.0.1:18889 \
  .venv/bin/python submit.py --recurring --cron '*/30 * * * *'
```

## Observable signals

Pipeline steps emit:

- spans named `kubeflow.pipeline.step <step>` plus client spans for Model
  Registry, KServe deployment, and prediction;
- `suse.ai.kubeflow.pipeline.step.runs` and
  `suse.ai.kubeflow.pipeline.step.duration`;
- `suse.ai.kubeflow.model.accuracy`;
- `suse.ai.kubeflow.deployment.smoke_test`.

The collector also scrapes the KFP API server and Argo workflow controller and
performs a synthetic Model Registry API check. In topology, verify these edges:

- `kubeflow-pipelines -> ml-registry.kubeflow`
- `kubeflow-pipelines -> inference-engine.kserve`
- `agent-service -> ml-registry.kubeflow`
- `agent-service -> inference-engine.kserve`

The agent's `[demo:list-models]`, `[demo:predict]`, and `[demo:lifecycle]`
messages provide deterministic demonstrations while normal natural-language
tool use remains enabled.

## Troubleshooting

- **KFP API returns an empty pipeline list:** direct multi-user API calls require
  `kubeflow-userid`; `submit.py` sets it on every KFP client API.
- **Deploy step gets 403:** check the RoleBinding generated by
  `kubeflowPipeline.rbac` and run `kubectl auth can-i` as the real profile
  service account.
- **KServe image pull fails:** attach `suse-ai-registry` to
  `suse-ai-kserve-model`; the deploy step preserves that pull secret.
- **Storage initializer fails:** confirm the KFP artifact URI exists and the
  generated `suse-ai-kserve-s3` Secret points to the Kubeflow S3-compatible
  object store.
- **Agent receives an OIDC redirect:** use the stable
  `suse-ai-sklearn-iris.<profile>.svc.cluster.local` URL, not the public KServe
  route.
- **Stable Service selects the previous revision:** the smoke step must wait for
  `latestCreatedRevision == latestReadyRevision`; do not bind from only
  `latestReadyRevision` during an active rollout.
