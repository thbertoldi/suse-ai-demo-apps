# Releasing the demo apps

Releases are independent of extension/Collector version numbers. Version 0.2.0
is the first consolidated release candidate; source preparation does not mean
its images or GitHub release have been published.

1. Prepare a PR from current main. Update chart `version` and `appVersion`, the
   default Iris pipeline image, README examples and `docs/releases/vVERSION.md`.
   Core/training image tags default to `appVersion`. Run `task validate` and
   review the PR's seven image builds and runtime checks.
2. Merge the reviewed PR. Require every main build/push job to succeed. Verify
   the candidate images using their `sha-COMMIT` tags or exact digests on a test
   namespace with the intended products. Include chat/RAG/agent traffic, every
   selected product scenario, training checkpoint roundtrip, and Kubeflow if
   enabled. Check fresh charts/traces, endpoint identity and native pod links.
   Record `sts topology-sync list` and relevant `describe` output in the correct
   Observability context. Keep private evidence and credentials outside Git.
3. Confirm the version tag/release does not already exist. From the reviewed
   main commit, create and push a new annotated `vVERSION` tag. Never move a
   published tag or reuse a version for changed content.
4. The tag workflow reruns validation, builds and smoke-tests all seven images,
   publishes versioned GHCR tags, packages the chart, and creates a **draft**
   GitHub release with the chart and checksums. Any failed image job blocks
   release creation. A failed publication can leave some image tags uploaded;
   retry the failed workflow at the unchanged tag, not a moved tag.
5. Test the tagged images/chart, including anonymous image pulls. Make every
   intended public GHCR package public; a public source repository alone does
   not make its packages public. Publish the draft only after acceptance:

   ```bash
   gh release edit v0.2.0 --draft=false
   ```

The image repositories are `ghcr.io/thbertoldi/suse-ai-demo-` followed by
`gateway`, `rag-service`, `llm-service`, `agent-service`, `traffic-gen`,
`iris-pipeline`, or `training`. Product traffic reuses `rag-service`.

## GHCR access prerequisite

An earlier Iris push failed with `permission_denied: write_package` despite the
workflow having `packages: write`. In the existing
[`suse-ai-demo-iris-pipeline` package settings](https://github.com/users/thbertoldi/packages/container/suse-ai-demo-iris-pipeline/settings),
check its linked repository and **Manage Actions access**. Grant
`thbertoldi/suse-ai-demo-apps` write access or enable the appropriate inherited
repository permissions. The package owner must perform this if the available
credentials cannot administer packages. Do not delete the existing package or
rename it to hide the failure. Then rerun the failed main publication job.

Images carry `org.opencontainers.image.source` to identify this repository.
GitHub documents package association and Actions access in
[Working with the Container registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).

## Local checks

`task validate` needs uv, Python 3.12, Go, Helm, Task, ShellCheck and actionlint.
It creates ignored `.venv-core`, `.venv-kubeflow` and `.venv-training` environments.
The split is intentional: the KFP image retains its tested OTel 1.40 dependencies.
The training environment downloads CPU-only PyTorch wheels.

Build an image with its service Dockerfile, then run
`bash scripts/smoke-image.sh SERVICE IMAGE`. Runtime checks catch missing packaged
modules; they do not certify connectivity to external services. The training
regression runs a real 200-epoch optimization and checks HTTP artifact roundtrip
and OTLP export against local protocol fixtures.
