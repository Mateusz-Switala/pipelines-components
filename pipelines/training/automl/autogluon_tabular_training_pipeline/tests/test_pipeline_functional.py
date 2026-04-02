"""Functional tests for AutoGluon tabular training pipeline (TC-A matrix).

These tests exercise the full end-to-end pipeline on a real RHOAI cluster with
real datasets. Each scenario is driven by a JSON config file (default:
functional_test_configs.json). Scenarios are parametrized so pytest runs each
independently.

Environment variables (on top of the standard integration .env):
    AUTOML_FUNCTIONAL_TEST_CONFIG  — path to the JSON config file
                              (default: functional_test_configs.json in this dir)
    AUTOML_FUNCTIONAL_TEST_REPORT  — path to the JSON report output
                              (default: functional_test_report.json in this dir)

Deployment test environment variables (optional, only when RHOAI_DEPLOY_AFTER_TRAINING=true):
    RHOAI_DEPLOY_AFTER_TRAINING    — set to "true"/"1" to run KServe deployment after pipeline
    RHOAI_SERVING_IMAGE            — container image for AutoGluon ServingRuntime (required)
    RHOAI_SERVING_RUNTIME_NAME     — ServingRuntime name (default: kserve-autogluonserver)
    RHOAI_CREATE_SERVING_RUNTIME   — set to "true"/"1" to create the ServingRuntime if missing
    RHOAI_INFERENCE_TIMEOUT        — seconds to wait for InferenceService ready (default: 300)

Parallel execution:
    Use pytest-xdist to run scenarios concurrently by passing ``-n <workers>``
    to pytest (e.g. ``-n 3``).

The test flow for each scenario:
    1. Read scenario configuration from the config file
    2. Upload the dataset to S3
    3. Run the pipeline with the correct input arguments
    4. Wait until the pipeline run completes successfully
    5. Measure wall-clock time for the run
    6. Read metrics for top_n models from the S3 artifacts
    7. Measure the total size of resulting model artifacts in S3
    8. [Optional] Deploy top-1 model via KServe and validate readiness / scoring
    9. Write the scenario result to a per-scenario JSON file
    10. Assert that the run completed in under 1 hour

After all scenarios finish, session-scoped teardown removes:
    - Uploaded datasets from S3
    - Pipeline artifacts from S3 (under the pipeline run prefix)

Report merging (xdist-safe):
    Each worker writes its scenario result to an individual file under
    .report_parts/{scenario_id}.json. A session-finish hook on the controller
    (or single-process run) merges them into the final report JSON.
"""

import json
import logging
import os
import re
import secrets
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TESTS_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _TESTS_DIR / "functional_test_configs.json"
_REPORT_PARTS_DIR = _TESTS_DIR / ".report_parts"
MAX_RUN_DURATION_SECONDS = 3600  # 1 hour hard limit

# ---------------------------------------------------------------------------
# KServe deployment constants
# ---------------------------------------------------------------------------

_KSERVE_GROUP = "serving.kserve.io"
_KSERVE_ISVC_VERSION = "v1beta1"
_KSERVE_SR_VERSION = "v1alpha1"
_KSERVE_ISVC_PLURAL = "inferenceservices"
_KSERVE_SR_PLURAL = "servingruntimes"

from ..pipeline import autogluon_tabular_training_pipeline  # noqa: E402

PIPELINE_DISPLAY_NAME = autogluon_tabular_training_pipeline.name


# ---------------------------------------------------------------------------
# Functional test config dataclass
# ---------------------------------------------------------------------------


@dataclass
class FunctionalTestConfig:
    """Configuration for a single functional test scenario."""

    id: str
    dataset_path: str
    label_column: str
    task_type: str
    top_n: int
    train_data_file_key: str
    tags: list[str] = field(default_factory=list)
    inference_sample: list[dict] | None = None  # v1 protocol instances list for post-training scoring

    def get_pipeline_arguments(
        self,
        train_data_bucket_name: str,
        train_data_secret_name: str,
    ) -> dict[str, Any]:
        """Build pipeline arguments dict for this scenario."""
        return {
            "train_data_secret_name": train_data_secret_name,
            "train_data_bucket_name": train_data_bucket_name,
            "train_data_file_key": self.train_data_file_key,
            "label_column": self.label_column,
            "task_type": self.task_type,
            "top_n": self.top_n,
        }


def _load_functional_configs(config_path: str | Path | None = None) -> list[FunctionalTestConfig]:
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    configs = []
    for item in raw:
        configs.append(
            FunctionalTestConfig(
                id=item["id"],
                dataset_path=item["dataset_path"],
                label_column=item["label_column"],
                task_type=item["task_type"],
                top_n=item["top_n"],
                train_data_file_key=item["train_data_file_key"],
                tags=item.get("tags", []),
                inference_sample=item.get("inference_sample"),
            )
        )
    return configs


# ---------------------------------------------------------------------------
# Load configs at collection time so parametrize works
# ---------------------------------------------------------------------------

_config_path_env = os.environ.get("AUTOML_FUNCTIONAL_TEST_CONFIG")
FUNCTIONAL_CONFIGS = _load_functional_configs(_config_path_env)


# ---------------------------------------------------------------------------
# Lazy integration config import (avoid import-guard issues)
# ---------------------------------------------------------------------------


def _session_rhoai_integration_config():
    from integration_config import RHOAI_INTEGRATION_CONFIG

    return RHOAI_INTEGRATION_CONFIG


RHOAI_INTEGRATION_CONFIG = _session_rhoai_integration_config()

# Read deployment flags after dotenv is loaded by _session_rhoai_integration_config().
DEPLOY_AFTER_TRAINING: bool = os.environ.get("RHOAI_DEPLOY_AFTER_TRAINING", "").strip().lower() in ("1", "true", "yes")
MAX_INFERENCE_READY_SECONDS: int = int(os.environ.get("RHOAI_INFERENCE_TIMEOUT", "300"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_name() -> str:
    hex_part = secrets.token_hex(3)
    time_part = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"automl-functional-{hex_part}-{time_part}"


def _run_succeeded(detail) -> bool:
    run = getattr(detail, "run", detail)
    state = getattr(run, "state", None)
    if state is None and hasattr(run, "status"):
        state = getattr(run.status, "state", None)
    if isinstance(state, str):
        return state.upper() == "SUCCEEDED"
    return False


def _list_s3_objects(s3_client, bucket: str, prefix: str) -> list[dict]:
    """List all objects under a prefix. Returns list of {Key, Size, ...} dicts."""
    paginator = s3_client.get_paginator("list_objects_v2")
    return [obj for page in paginator.paginate(Bucket=bucket, Prefix=prefix) for obj in page.get("Contents", [])]


def _read_s3_json(s3_client, bucket: str, key: str) -> dict | None:
    """Read and parse a JSON file from S3. Returns None on failure."""
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except Exception as e:
        logger.warning("Failed to read s3://%s/%s: %s", bucket, key, e)
        return None


def _collect_model_metrics_and_sizes(s3_client, bucket: str, run_prefix: str) -> list[dict]:
    """Scan S3 artifacts for metrics.json files and compute per-model predictor size.

    Returns:
        list of dicts with keys:
            {model_name, metrics, artifact_key, total_predictor_size_bytes, total_predictor_size_mb}
    """
    objects = _list_s3_objects(s3_client, bucket, run_prefix)

    # Collect metrics entries keyed by model name
    metrics_by_model: dict[str, dict] = {}
    for obj in objects:
        key = obj["Key"]
        if key.endswith("metrics.json") and "/metrics/metrics.json" in key:
            data = _read_s3_json(s3_client, bucket, key)
            if data is not None:
                # Path: .../ModelName_FULL/metrics/metrics.json → go 2 levels up
                parts = key.rsplit("/", 3)
                model_name = parts[-3] if len(parts) >= 3 else "unknown"
                metrics_by_model[model_name] = {
                    "model_name": model_name,
                    "metrics": data,
                    "artifact_key": key,
                    "total_predictor_size_bytes": 0,
                }

    # Sum sizes of objects under <ModelName>/predictor/ for each model
    for obj in objects:
        key = obj["Key"]
        for model_name in metrics_by_model:
            if f"/{model_name}/predictor/" in key:
                metrics_by_model[model_name]["total_predictor_size_bytes"] += obj.get("Size", 0)
                break

    for entry in metrics_by_model.values():
        entry["total_predictor_size_mb"] = round(entry["total_predictor_size_bytes"] / (1024 * 1024), 2)

    return list(metrics_by_model.values())


def _find_top_model_predictor_prefix(s3_client, bucket: str, run_prefix: str, model_name: str) -> str | None:
    """Find the S3 prefix (no trailing slash) for a model's predictor directory.

    Scans objects under ``run_prefix`` and returns the key prefix that ends at
    ``/<model_name>/predictor`` (exclusive of any trailing slash).
    Returns None if no matching object is found.
    """
    objects = _list_s3_objects(s3_client, bucket, run_prefix)
    needle = f"/{model_name}/predictor/"
    for obj in objects:
        key = obj["Key"]
        idx = key.find(needle)
        if idx != -1:
            return key[: idx + len(needle) - 1]  # strip trailing slash
    return None


# ---------------------------------------------------------------------------
# KServe resource helpers
# ---------------------------------------------------------------------------


def _make_isvc_name(scenario_id: str, run_id: str) -> str:
    """Return a valid Kubernetes name for an InferenceService."""
    clean = re.sub(r"[^a-z0-9]+", "-", scenario_id.lower()).strip("-")[:40]
    return f"automl-{clean}-{run_id[:8]}"


def _load_k8s_config(kubeconfig_path: str | None) -> None:
    """Load kubernetes config from file or fall back to in-cluster config."""
    from kubernetes import config

    try:
        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
        else:
            config.load_kube_config()
    except Exception:
        config.load_incluster_config()


def _create_kserve_s3_secret(v1, namespace: str, secret_name: str, integration_config: dict) -> None:
    """Create (or replace) a plain S3 credentials secret for KServe storage initializer.

    Uses ``spec.predictor.model.storage.key`` instead of the SA-annotation mechanism,
    which is incompatible with RHOAI's ``automountServiceAccountToken: false`` admission
    controller injection. The storage initializer reads these keys directly as env vars.
    """
    from kubernetes import client
    from kubernetes.client.rest import ApiException

    endpoint = integration_config["s3_endpoint"]
    region = integration_config.get("s3_region", "us-east-1")

    secret = client.V1Secret(
        metadata=client.V1ObjectMeta(name=secret_name, namespace=namespace),
        type="Opaque",
        string_data={
            "AWS_ACCESS_KEY_ID": integration_config["s3_access_key"],
            "AWS_SECRET_ACCESS_KEY": integration_config["s3_secret_key"],
            "AWS_S3_ENDPOINT": endpoint,
            "AWS_DEFAULT_REGION": region,
        },
    )
    try:
        v1.create_namespaced_secret(namespace, secret)
    except ApiException as e:
        if e.status == 409:
            v1.replace_namespaced_secret(secret_name, namespace, secret)
        else:
            raise


def _ensure_serving_runtime(co, namespace: str, runtime_name: str, serving_image: str) -> None:
    """Create the AutoGluon ServingRuntime if it does not exist in the namespace."""
    from kubernetes.client.rest import ApiException

    try:
        co.get_namespaced_custom_object(
            group=_KSERVE_GROUP,
            version=_KSERVE_SR_VERSION,
            namespace=namespace,
            plural=_KSERVE_SR_PLURAL,
            name=runtime_name,
        )
        logger.info("ServingRuntime %r already exists in %r — skipping creation", runtime_name, namespace)
        return
    except ApiException as e:
        if e.status != 404:
            raise

    runtime = {
        "apiVersion": f"{_KSERVE_GROUP}/{_KSERVE_SR_VERSION}",
        "kind": "ServingRuntime",
        "metadata": {
            "name": runtime_name,
            "namespace": namespace,
            "annotations": {"openshift.io/display-name": "AutoGluon ServingRuntime for KServe"},
        },
        "spec": {
            "annotations": {
                "prometheus.kserve.io/port": "8080",
                "prometheus.kserve.io/path": "/metrics",
            },
            "supportedModelFormats": [{"name": "autogluon", "version": "1"}],
            "protocolVersions": ["v1", "v2"],
            "containers": [
                {
                    "name": "kserve-container",
                    "image": serving_image,
                    "args": [
                        "--model_name={{.Name}}",
                        "--model_dir=/mnt/models",
                        "--http_port=8080",
                    ],
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "privileged": False,
                        "runAsNonRoot": True,
                        "capabilities": {"drop": ["ALL"]},
                    },
                    "resources": {
                        "requests": {"cpu": "1", "memory": "2Gi"},
                        "limits": {"cpu": "1", "memory": "2Gi"},
                    },
                }
            ],
        },
    }
    co.create_namespaced_custom_object(
        group=_KSERVE_GROUP,
        version=_KSERVE_SR_VERSION,
        namespace=namespace,
        plural=_KSERVE_SR_PLURAL,
        body=runtime,
    )
    logger.info("Created ServingRuntime %r in %r", runtime_name, namespace)


def _create_inference_service(
    co,
    namespace: str,
    isvc_name: str,
    runtime_name: str,
    storage_uri: str,
    secret_name: str,
) -> None:
    """Create a KServe InferenceService in RawDeployment mode.

    Uses ``spec.predictor.model.storage.key`` to pass S3 credentials directly,
    bypassing the SA-annotation mechanism which is broken on RHOAI because the
    admission controller injects ``automountServiceAccountToken: false``.
    """
    from kubernetes.client.rest import ApiException

    isvc = {
        "apiVersion": f"{_KSERVE_GROUP}/{_KSERVE_ISVC_VERSION}",
        "kind": "InferenceService",
        "metadata": {
            "name": isvc_name,
            "namespace": namespace,
            "annotations": {"serving.kserve.io/deploymentMode": "RawDeployment"},
        },
        "spec": {
            "predictor": {
                "model": {
                    "modelFormat": {"name": "autogluon", "version": "1"},
                    "runtime": runtime_name,
                    "storageUri": storage_uri,
                    "storage": {"key": secret_name},
                },
            }
        },
    }
    try:
        co.create_namespaced_custom_object(
            group=_KSERVE_GROUP,
            version=_KSERVE_ISVC_VERSION,
            namespace=namespace,
            plural=_KSERVE_ISVC_PLURAL,
            body=isvc,
        )
    except ApiException as e:
        if e.status != 409:
            raise


def _wait_for_inference_service_ready(
    co, namespace: str, isvc_name: str, timeout_seconds: int
) -> tuple[bool, str | None]:
    """Poll the InferenceService until Ready=True or timeout.

    Returns ``(ready, url)`` where ``url`` is ``status.url`` when the ISVC is
    ready, or ``None`` on timeout.
    """
    from kubernetes.client.rest import ApiException

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            isvc = co.get_namespaced_custom_object(
                group=_KSERVE_GROUP,
                version=_KSERVE_ISVC_VERSION,
                namespace=namespace,
                plural=_KSERVE_ISVC_PLURAL,
                name=isvc_name,
            )
        except ApiException:
            time.sleep(10)
            continue

        status = isvc.get("status") or {}
        conditions = status.get("conditions") or []
        for c in conditions:
            if c.get("type") == "Ready" and c.get("status") == "True":
                return True, status.get("url")

        time.sleep(15)

    return False, None


def _get_isvc_external_url(co, namespace: str, isvc_name: str) -> str | None:
    """Look up an OpenShift Route with the same name as the InferenceService.

    KServe in RawDeployment mode on OpenShift creates a Route named after the
    InferenceService. Returns ``https://<host>`` or None if not found.
    """
    from kubernetes.client.rest import ApiException

    try:
        route = co.get_namespaced_custom_object(
            group="route.openshift.io",
            version="v1",
            namespace=namespace,
            plural="routes",
            name=isvc_name,
        )
        host = (route.get("spec") or {}).get("host")
        if not host:
            ingress = (route.get("status") or {}).get("ingress") or []
            host = ingress[0].get("host") if ingress else None
        return f"https://{host}" if host else None
    except ApiException:
        return None


def _score_inference_service(isvc_url: str, model_name: str, instances: list[dict], token: str | None) -> dict:
    """Send a KServe v1 predict request and return the parsed JSON response."""
    import urllib.request

    url = f"{isvc_url.rstrip('/')}/v1/models/{model_name}:predict"
    payload = json.dumps({"instances": instances}).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    # NOTE: TLS verification is skipped via an unverified SSL context for test clusters.
    import ssl

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        return json.loads(resp.read().decode())


def _delete_inference_service(co, namespace: str, isvc_name: str) -> None:
    """Delete an InferenceService, silently ignoring 404."""
    from kubernetes.client.rest import ApiException

    try:
        co.delete_namespaced_custom_object(
            group=_KSERVE_GROUP,
            version=_KSERVE_ISVC_VERSION,
            namespace=namespace,
            plural=_KSERVE_ISVC_PLURAL,
            name=isvc_name,
        )
    except ApiException as e:
        if e.status != 404:
            logger.warning("Failed to delete InferenceService %r: %s", isvc_name, e)


def _delete_s3_objects(s3_client, bucket: str, keys: list[str]) -> int:
    """Delete objects from S3 in batches. Returns count of deleted objects."""
    deleted = 0
    batch_size = 1000
    for i in range(0, len(keys), batch_size):
        batch = keys[i : i + batch_size]
        delete_req = {"Objects": [{"Key": k} for k in batch], "Quiet": True}
        try:
            s3_client.delete_objects(Bucket=bucket, Delete=delete_req)
            deleted += len(batch)
        except Exception as e:
            logger.warning("Failed to delete %d objects from s3://%s: %s", len(batch), bucket, e)
    return deleted


# ---------------------------------------------------------------------------
# Per-scenario report file I/O (xdist-safe: each worker writes its own file)
# ---------------------------------------------------------------------------


def _write_scenario_result(scenario_id: str, result: dict) -> None:
    """Write a single scenario result to .report_parts/{scenario_id}.json."""
    _REPORT_PARTS_DIR.mkdir(parents=True, exist_ok=True)
    part_path = _REPORT_PARTS_DIR / f"{scenario_id}.json"
    part_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Session-scoped fixtures for S3 cleanup tracking
# ---------------------------------------------------------------------------


class S3CleanupTracker:
    """Accumulates S3 keys to delete during session teardown."""

    def __init__(self):
        """Initialize with empty tracking dicts."""
        self.uploaded_dataset_keys: dict[str, list[str]] = {}  # bucket -> [keys]
        self.artifact_prefixes: dict[str, list[str]] = {}  # bucket -> [prefixes]

    def track_upload(self, bucket: str, key: str) -> None:
        """Record an uploaded dataset key for teardown cleanup."""
        self.uploaded_dataset_keys.setdefault(bucket, []).append(key)

    def track_artifact_prefix(self, bucket: str, prefix: str) -> None:
        """Record a pipeline artifact prefix for teardown cleanup."""
        self.artifact_prefixes.setdefault(bucket, []).append(prefix)


@pytest.fixture(scope="session")
def s3_cleanup_tracker():
    """Session-scoped S3 cleanup tracker shared across all scenarios."""
    return S3CleanupTracker()


@pytest.fixture(scope="session", autouse=True)
def s3_teardown(s3_client, s3_cleanup_tracker):
    """Session-scoped teardown: delete uploaded datasets and pipeline artifacts from S3."""
    yield
    if s3_client is None:
        return

    logger.info("Starting S3 cleanup...")

    # Delete uploaded datasets
    for bucket, keys in s3_cleanup_tracker.uploaded_dataset_keys.items():
        if keys:
            count = _delete_s3_objects(s3_client, bucket, keys)
            logger.info("Deleted %d uploaded dataset objects from s3://%s", count, bucket)

    # Delete artifact objects under tracked prefixes (unless AUTOML_FUNCTIONAL_TEST_KEEP_ARTIFACTS is set)
    keep_artifacts = os.environ.get("AUTOML_FUNCTIONAL_TEST_KEEP_ARTIFACTS", "").strip().lower() in ("1", "true", "yes")
    if keep_artifacts:
        logger.info("AUTOML_FUNCTIONAL_TEST_KEEP_ARTIFACTS is set — skipping artifact cleanup")
    for bucket, prefixes in s3_cleanup_tracker.artifact_prefixes.items():
        if keep_artifacts:
            break
        for prefix in prefixes:
            objects = _list_s3_objects(s3_client, bucket, prefix)
            if objects:
                keys = [o["Key"] for o in objects]
                count = _delete_s3_objects(s3_client, bucket, keys)
                logger.info("Deleted %d artifact objects from s3://%s/%s", count, bucket, prefix)

    logger.info("S3 cleanup complete.")


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.functional
@pytest.mark.skipif(
    RHOAI_INTEGRATION_CONFIG is None,
    reason=("RHOAI integration env not set (set RHOAI_URL, RHOAI_TOKEN, S3 vars; see .env.template)"),
)
@pytest.mark.parametrize(
    "func_config",
    FUNCTIONAL_CONFIGS,
    ids=[c.id for c in FUNCTIONAL_CONFIGS],
)
class TestAutogluonPipelineFunctional:
    """Functional tests running TC-A scenarios on a real RHOAI cluster."""

    def test_scenario(
        self,
        func_config: FunctionalTestConfig,
        rhoai_integration_config,
        rhoai_project,  # noqa: ARG002 — ensures project/secret exist before runs
        kfp_client,
        compiled_pipeline_path,
        s3_client,
        s3_cleanup_tracker: S3CleanupTracker,
        temp_kubeconfig_path,
    ):
        """Run one TC-A scenario end-to-end and validate results."""
        if not kfp_client:
            pytest.skip("KFP client not available")

        config = rhoai_integration_config
        data_bucket = config["s3_bucket_data"]
        artifacts_bucket = config.get("s3_bucket_artifacts") or data_bucket
        secret_name = config["s3_secret_name"]

        # ------------------------------------------------------------------
        # 1. Upload dataset to S3
        # ------------------------------------------------------------------
        dataset_local_path = _TESTS_DIR / func_config.dataset_path
        if not dataset_local_path.is_file():
            pytest.fail(f"Dataset file not found: {dataset_local_path}")

        # Count rows and features from the CSV header
        import csv

        with open(dataset_local_path, newline="", encoding="utf-8") as csvf:
            reader = csv.reader(csvf)
            header = next(reader)
            num_columns = len(header)
            num_rows = sum(1 for _ in reader)
        num_features = num_columns - 1  # exclude label column

        logger.info(
            "Dataset %s: %d rows, %d features (+ label '%s')",
            dataset_local_path.name,
            num_rows,
            num_features,
            func_config.label_column,
        )

        s3_key = func_config.train_data_file_key
        logger.info("Uploading %s -> s3://%s/%s", dataset_local_path.name, data_bucket, s3_key)
        s3_client.put_object(
            Bucket=data_bucket,
            Key=s3_key,
            Body=dataset_local_path.read_bytes(),
            ContentType="text/csv",
        )
        s3_cleanup_tracker.track_upload(data_bucket, s3_key)

        # ------------------------------------------------------------------
        # 2. Build pipeline arguments and submit run
        # ------------------------------------------------------------------
        arguments = func_config.get_pipeline_arguments(
            train_data_bucket_name=data_bucket,
            train_data_secret_name=secret_name,
        )
        run_name = _make_run_name()
        logger.info(
            "Submitting pipeline run %s for scenario %s (task_type=%s, top_n=%d)",
            run_name,
            func_config.id,
            func_config.task_type,
            func_config.top_n,
        )

        start_time = time.monotonic()
        start_ts = datetime.now(timezone.utc)

        run = kfp_client.create_run_from_pipeline_package(
            compiled_pipeline_path,
            arguments=arguments,
            run_name=run_name,
        )
        run_id = run.run_id

        # ------------------------------------------------------------------
        # 3. Wait for pipeline completion
        # ------------------------------------------------------------------
        timeout = int(os.environ.get("RHOAI_PIPELINE_RUN_TIMEOUT", str(MAX_RUN_DURATION_SECONDS)))
        detail = kfp_client.wait_for_run_completion(run_id, timeout=timeout)

        elapsed_seconds = time.monotonic() - start_time
        elapsed_minutes = elapsed_seconds / 60.0

        logger.info("Run %s completed in %.1f min (%.0f s)", run_id, elapsed_minutes, elapsed_seconds)

        # ------------------------------------------------------------------
        # 4. Assert pipeline succeeded
        # ------------------------------------------------------------------
        succeeded = _run_succeeded(detail)
        assert succeeded, (
            f"Pipeline run {run_id} did not succeed for scenario {func_config.id}; "
            f"state={getattr(getattr(detail, 'run', detail), 'state', 'unknown')}"
        )

        # ------------------------------------------------------------------
        # 5. Read metrics and measure model sizes from S3 artifacts
        # ------------------------------------------------------------------
        run_prefix = f"{PIPELINE_DISPLAY_NAME}/{run_id}"
        s3_cleanup_tracker.track_artifact_prefix(artifacts_bucket, run_prefix)

        metrics_list = _collect_model_metrics_and_sizes(s3_client, artifacts_bucket, run_prefix)

        logger.info(
            "Scenario %s: found %d models",
            func_config.id,
            len(metrics_list),
        )
        for m in metrics_list:
            logger.info(
                "  Model: %s | Predictor: %.2f MB | Metrics: %s",
                m["model_name"],
                m["total_predictor_size_mb"],
                json.dumps(m["metrics"], default=str),
            )

        # ------------------------------------------------------------------
        # 6. [Optional] Deploy top-1 model via KServe and validate
        # ------------------------------------------------------------------
        deployment_result: dict = {}
        if DEPLOY_AFTER_TRAINING and metrics_list:
            deployment_result = self._run_deployment_test(
                func_config=func_config,
                metrics_list=metrics_list,
                s3_client=s3_client,
                artifacts_bucket=artifacts_bucket,
                run_prefix=run_prefix,
                rhoai_integration_config=config,
                temp_kubeconfig_path=temp_kubeconfig_path,
            )

        # ------------------------------------------------------------------
        # 7. Write per-scenario report (xdist-safe: one file per scenario)
        # ------------------------------------------------------------------
        _write_scenario_result(
            func_config.id,
            {
                "run_id": run_id,
                "run_name": run_name,
                "started_at": start_ts.isoformat(),
                "elapsed_seconds": round(elapsed_seconds, 1),
                "elapsed_minutes": round(elapsed_minutes, 2),
                "succeeded": succeeded,
                "task_type": func_config.task_type,
                "top_n": func_config.top_n,
                "label_column": func_config.label_column,
                "train_data_file_key": func_config.train_data_file_key,
                "dataset_rows": num_rows,
                "dataset_features": num_features,
                "models": [
                    {
                        "model_name": m["model_name"],
                        "metrics": m["metrics"],
                        "total_predictor_size_bytes": m["total_predictor_size_bytes"],
                        "total_predictor_size_mb": m["total_predictor_size_mb"],
                    }
                    for m in metrics_list
                ],
                "deployment": deployment_result,
                "config": asdict(func_config),
            },
        )

        # ------------------------------------------------------------------
        # 8. Assertions
        # ------------------------------------------------------------------
        assert elapsed_seconds < MAX_RUN_DURATION_SECONDS, (
            f"Scenario {func_config.id} took {elapsed_minutes:.1f} min "
            f"({elapsed_seconds:.0f} s), exceeding the 1-hour limit"
        )

        assert len(metrics_list) >= 1, (
            f"Expected at least 1 model with metrics for scenario {func_config.id}; "
            f"found {len(metrics_list)} under s3://{artifacts_bucket}/{run_prefix}"
        )

        if DEPLOY_AFTER_TRAINING:
            assert deployment_result.get("isvc_ready"), (
                f"InferenceService for scenario {func_config.id} did not become ready within "
                f"{MAX_INFERENCE_READY_SECONDS}s. Details: {deployment_result}"
            )

    # ------------------------------------------------------------------
    # KServe deployment helper (called from test_scenario when enabled)
    # ------------------------------------------------------------------

    def _run_deployment_test(
        self,
        *,
        func_config: "FunctionalTestConfig",
        metrics_list: list[dict],
        s3_client,
        artifacts_bucket: str,
        run_prefix: str,
        rhoai_integration_config: dict,
        temp_kubeconfig_path: str | None,
    ) -> dict:
        """Deploy the top-1 model from a completed pipeline run via KServe.

        Steps:
          a. Find the top-1 model's predictor prefix in S3.
          b. Create a plain S3 credentials secret (referenced via storage.key).
          c. Optionally create the ServingRuntime (if RHOAI_CREATE_SERVING_RUNTIME=true).
          d. Create the InferenceService with storage.key pointing at the secret.
          e. Wait for the InferenceService to become ready.
          f. Score the model if ``func_config.inference_sample`` is provided.
          g. Clean up the InferenceService.

        Returns a dict that is merged into the scenario report.
        """
        try:
            from kubernetes import client
        except ImportError:
            logger.warning("kubernetes package not installed; skipping deployment test (pip install kubernetes)")
            return {"skipped": True, "reason": "kubernetes package not installed"}

        namespace = rhoai_integration_config["rhoai_project"]
        token = rhoai_integration_config.get("rhoai_token")
        serving_runtime_name = os.environ.get("RHOAI_SERVING_RUNTIME_NAME", "kserve-autogluonserver")
        serving_image = os.environ.get("RHOAI_SERVING_IMAGE", "").strip()
        create_runtime = os.environ.get("RHOAI_CREATE_SERVING_RUNTIME", "").strip().lower() in ("1", "true", "yes")

        # Pick top-1 model (first entry returned by _collect_model_metrics_and_sizes)
        top_model = metrics_list[0]
        model_name = top_model["model_name"]

        result: dict = {
            "model_name": model_name,
            "serving_runtime": serving_runtime_name,
            "isvc_ready": False,
            "isvc_url": None,
            "scored": False,
            "predictions": None,
        }

        # a. Locate the predictor directory in S3
        predictor_prefix = _find_top_model_predictor_prefix(s3_client, artifacts_bucket, run_prefix, model_name)
        if predictor_prefix is None:
            logger.warning(
                "Could not find predictor prefix for model %r under %s/%s", model_name, artifacts_bucket, run_prefix
            )
            result["error"] = f"Predictor prefix not found for model {model_name!r}"
            return result

        storage_uri = f"s3://{artifacts_bucket}/{predictor_prefix}"
        result["storage_uri"] = storage_uri
        logger.info("Deploying model %r from %s", model_name, storage_uri)

        isvc_name = _make_isvc_name(func_config.id, run_prefix.split("/")[-1])
        kserve_secret_name = f"kserve-s3-{isvc_name[:40]}"
        result["isvc_name"] = isvc_name
        isvc_created = False

        try:
            _load_k8s_config(temp_kubeconfig_path)
            v1 = client.CoreV1Api()
            co = client.CustomObjectsApi()

            # b. S3 credentials secret (referenced directly via storage.key in the ISVC spec)
            _create_kserve_s3_secret(v1, namespace, kserve_secret_name, rhoai_integration_config)

            # c. ServingRuntime (optional creation)
            if create_runtime:
                if not serving_image:
                    logger.warning(
                        "RHOAI_CREATE_SERVING_RUNTIME=true but RHOAI_SERVING_IMAGE is not set; skipping runtime creation"  # noqa: E501
                    )
                else:
                    _ensure_serving_runtime(co, namespace, serving_runtime_name, serving_image)

            # d. Create InferenceService (credentials via storage.key, no serviceAccountName needed)
            _create_inference_service(co, namespace, isvc_name, serving_runtime_name, storage_uri, kserve_secret_name)
            isvc_created = True
            logger.info("Created InferenceService %r in namespace %r", isvc_name, namespace)

            # e. Wait for ready
            ready, isvc_url = _wait_for_inference_service_ready(co, namespace, isvc_name, MAX_INFERENCE_READY_SECONDS)
            result["isvc_ready"] = ready

            if not ready:
                logger.warning(
                    "InferenceService %r did not become ready within %ds", isvc_name, MAX_INFERENCE_READY_SECONDS
                )
                return result

            # Prefer external Route URL for scoring (allows access from outside the cluster)
            external_url = _get_isvc_external_url(co, namespace, isvc_name)
            effective_url = external_url or isvc_url
            result["isvc_url"] = effective_url
            logger.info("InferenceService %r is ready. URL: %s", isvc_name, effective_url)

            # f. Score if inference_sample provided
            if func_config.inference_sample and effective_url:
                try:
                    response = _score_inference_service(effective_url, isvc_name, func_config.inference_sample, token)
                    result["scored"] = True
                    result["predictions"] = response.get("predictions")
                    logger.info("Scoring response for %r: %s", isvc_name, json.dumps(response, default=str))
                except Exception as score_err:
                    logger.warning("Scoring request failed for %r: %s", isvc_name, score_err)
                    result["score_error"] = str(score_err)
            elif not func_config.inference_sample:
                logger.info("No inference_sample in config %r — skipping scoring", func_config.id)

        except Exception as deploy_err:
            logger.error("Deployment test failed for scenario %r: %s", func_config.id, deploy_err, exc_info=True)
            result["error"] = str(deploy_err)

        finally:
            # g. Always clean up the InferenceService (only if it was successfully created)
            if isvc_created:
                try:
                    _load_k8s_config(temp_kubeconfig_path)
                    co_cleanup = client.CustomObjectsApi()
                    _delete_inference_service(co_cleanup, namespace, isvc_name)
                    logger.info("Deleted InferenceService %r", isvc_name)
                except Exception as cleanup_err:
                    logger.warning("Failed to clean up InferenceService %r: %s", isvc_name, cleanup_err)

        return result
