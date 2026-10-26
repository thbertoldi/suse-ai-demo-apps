"""Version, submit, optionally schedule, and verify the Iris KFP demo."""

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import kfp
from kfp_server_api.exceptions import ApiException

from pipeline import compile_pipeline


def _set_user_header(client: kfp.Client, user_id: str) -> None:
    """Authenticate direct, port-forwarded calls to a multi-user KFP API."""
    if not user_id:
        return
    for name in (
        "_experiment_api",
        "_healthz_api",
        "_pipelines_api",
        "_recurring_run_api",
        "_run_api",
        "_upload_api",
    ):
        api = getattr(client, name, None)
        if api is not None:
            api.api_client.default_headers["kubeflow-userid"] = user_id


def _id(resource, *names: str) -> str:
    for name in names:
        value = getattr(resource, name, None)
        if value:
            return str(value)
    raise RuntimeError(f"KFP response did not contain any of {names!r}: {resource!r}")


def _get_or_create_experiment(
    client: kfp.Client,
    name: str,
    namespace: str,
):
    try:
        return client.get_experiment(experiment_name=name, namespace=namespace)
    except ApiException as exc:
        if exc.status != 404:
            raise
        return client.create_experiment(
            name=name,
            description="SUSE AI Observability end-to-end demo",
            namespace=namespace,
        )


def _upload_versioned_pipeline(
    client: kfp.Client,
    package: Path,
    pipeline_name: str,
    version_name: str,
    namespace: str,
) -> tuple[str, str]:
    # ``Client.get_pipeline_id`` does not pass a namespace to the KFP v2 API,
    # so it cannot find private pipelines in multi-user mode.  List the
    # profile namespace explicitly and match the display name instead.
    response = client.list_pipelines(page_size=100, namespace=namespace)
    matches = [
        item
        for item in (getattr(response, "pipelines", None) or [])
        if getattr(item, "display_name", "") == pipeline_name
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple pipelines named {pipeline_name!r} exist in {namespace!r}"
        )
    pipeline_id = _id(matches[0], "pipeline_id", "id") if matches else None
    if not pipeline_id:
        uploaded = client.upload_pipeline(
            pipeline_package_path=str(package),
            pipeline_name=pipeline_name,
            description="Real Iris train/register/deploy observability demo",
            namespace=namespace,
        )
        pipeline_id = _id(uploaded, "pipeline_id", "id")

    version = client.upload_pipeline_version(
        pipeline_package_path=str(package),
        pipeline_version_name=version_name,
        pipeline_id=pipeline_id,
        description=f"SUSE AI observability demo {version_name}",
    )
    return pipeline_id, _id(version, "pipeline_version_id", "version_id", "id")


def _find_recurring_run(client: kfp.Client, experiment_id: str, name: str):
    response = client.list_recurring_runs(
        experiment_id=experiment_id,
        page_size=100,
    )
    for recurring_run in getattr(response, "recurring_runs", None) or []:
        if getattr(recurring_run, "display_name", "") == name:
            return recurring_run
    return None


def _state(run) -> str:
    runtime_state = getattr(getattr(run, "state", None), "runtime_state", None)
    return str(runtime_state or getattr(run, "state", "")).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.environ.get("KFP_HOST", "http://127.0.0.1:18889"),
    )
    parser.add_argument(
        "--namespace",
        default=os.environ.get("KFP_NAMESPACE", "kubeflow-user-example-com"),
    )
    parser.add_argument(
        "--user-id",
        default=os.environ.get("KFP_USER_ID", "user@example.com"),
    )
    parser.add_argument(
        "--service-account",
        default=os.environ.get("KFP_SERVICE_ACCOUNT", "default-editor"),
    )
    parser.add_argument("--experiment", default="SUSE AI Observability Demo")
    parser.add_argument("--pipeline-name", default="iris-lifecycle")
    parser.add_argument("--run-name", default="iris-lifecycle-observability-demo")
    parser.add_argument("--version-name", default="")
    parser.add_argument("--minimum-accuracy", type=float, default=0.85)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument(
        "--recurring",
        action="store_true",
        help="Create a recurring run if one with the requested name is absent.",
    )
    parser.add_argument(
        "--recurring-name",
        default="iris-lifecycle-observability-refresh",
    )
    parser.add_argument("--cron", default="*/30 * * * *")
    args = parser.parse_args()

    version_name = args.version_name or datetime.now(timezone.utc).strftime(
        "demo-%Y%m%d-%H%M%S"
    )
    parameters = {
        "model_version": version_name,
        "minimum_accuracy": args.minimum_accuracy,
        "target_namespace": args.namespace,
    }

    with tempfile.TemporaryDirectory(prefix="suse-ai-kfp-") as directory:
        package = Path(directory) / "iris_pipeline.yaml"
        compile_pipeline(str(package))

        client = kfp.Client(host=args.host, namespace=args.namespace)
        _set_user_header(client, args.user_id)
        experiment = _get_or_create_experiment(
            client,
            args.experiment,
            args.namespace,
        )
        experiment_id = _id(experiment, "experiment_id", "id")
        pipeline_id, version_id = _upload_versioned_pipeline(
            client,
            package,
            args.pipeline_name,
            version_name,
            args.namespace,
        )
        print(
            f"uploaded pipeline {pipeline_id}, version {version_id} "
            f"({version_name})"
        )

        run = client.run_pipeline(
            experiment_id=experiment_id,
            job_name=f"{args.run_name}-{version_name}",
            pipeline_id=pipeline_id,
            version_id=version_id,
            params=parameters,
            enable_caching=False,
            service_account=args.service_account,
        )
        run_id = _id(run, "run_id", "id")
        print(f"submitted run {run_id}")

        if args.recurring:
            existing = _find_recurring_run(
                client,
                experiment_id,
                args.recurring_name,
            )
            if existing:
                print(
                    "recurring run already exists: "
                    + _id(existing, "recurring_run_id", "job_id", "id")
                )
            else:
                recurring = client.create_recurring_run(
                    experiment_id=experiment_id,
                    job_name=args.recurring_name,
                    cron_expression=args.cron,
                    no_catchup=True,
                    params=parameters,
                    pipeline_id=pipeline_id,
                    version_id=version_id,
                    enable_caching=False,
                    service_account=args.service_account,
                )
                print(
                    "created recurring run "
                    + _id(recurring, "recurring_run_id", "job_id", "id")
                )

        if not args.no_wait:
            completed = client.wait_for_run_completion(run_id, timeout=args.timeout)
            state = _state(completed)
            print(f"run {run_id} finished with state {state}")
            if state != "succeeded":
                raise SystemExit(1)


if __name__ == "__main__":
    main()
