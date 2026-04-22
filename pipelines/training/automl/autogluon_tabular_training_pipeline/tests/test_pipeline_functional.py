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
    RHOAI_SERVING_IMAGE            — container image for AutoGluon ServingRuntime (required when
                                     RHOAI_CREATE_SERVING_RUNTIME=true)
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
import ssl
import time
import urllib.error
import urllib.request
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

# Expected primary metric key per task type (used in metrics content assertions).
# AutoGluon always includes these when evaluate_predictions is called.
_TASK_PRIMARY_METRICS: dict[str, str] = {
    "regression": "r2",
    "binary": "accuracy",
    "multiclass": "accuracy",
}

# KServe API constants
_KSERVE_GROUP = "serving.kserve.io"
_KSERVE_ISVC_VERSION = "v1beta1"
_KSERVE_SR_VERSION = "v1alpha1"
_KSERVE_ISVC_PLURAL = "inferenceservices"
_KSERVE_SR_PLURAL = "servingruntimes"

# Timeout for each Kubernetes client call (avoids indefinite hang on unreachable API).
_K8S_CALL_TIMEOUT = 30  # seconds
# HardwareProfile GET can fail briefly after RBAC / operator startup; retry before giving up.
_HW_PROFILE_FETCH_ATTEMPTS = 6
_HW_PROFILE_FETCH_DELAY_SECONDS = 3.0

# Timeout for notebook execution via Kubernetes Job (overrideable via RHOAI_NOTEBOOK_RUN_TIMEOUT).
_NOTEBOOK_JOB_TIMEOUT = 1200  # seconds

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
    # Column-oriented sample for post-training inference scoring:
    # [{col: [val, ...], ...}] — converted to row-oriented instances before sending.
    inference_sample: list[dict] | None = None

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
    import importlib.util

    spec = importlib.util.spec_from_file_location("integration_config", _TESTS_DIR / "integration_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RHOAI_INTEGRATION_CONFIG


RHOAI_INTEGRATION_CONFIG = _session_rhoai_integration_config()

# Read deployment flags after dotenv is loaded by _session_rhoai_integration_config().
DEPLOY_AFTER_TRAINING: bool = os.environ.get("RHOAI_DEPLOY_AFTER_TRAINING", "").strip().lower() in (
    "1",
    "true",
    "yes",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_name() -> str:
    hex_part = secrets.token_hex(3)
    time_part = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"automl-tabular-functional-{hex_part}-{time_part}"


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
            {model_name, metrics, artifact_key, predictor_s3_uri,
             total_predictor_size_bytes, total_predictor_size_mb, notebook_key}
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
                # Derive predictor S3 URI from the metrics key
                model_dir_key = key.rsplit("/", 2)[0]  # .../ModelName
                predictor_s3_uri = f"s3://{bucket}/{model_dir_key}/predictor"
                metrics_by_model[model_name] = {
                    "model_name": model_name,
                    "metrics": data,
                    "artifact_key": key,
                    "predictor_s3_uri": predictor_s3_uri,
                    "total_predictor_size_bytes": 0,
                    "notebook_key": None,
                }

    # Sum predictor sizes and record notebook keys in a single pass (no early break so
    # both conditions are evaluated for every object against every model).
    for obj in objects:
        key = obj["Key"]
        size = obj.get("Size", 0)
        for model_name, entry in metrics_by_model.items():
            if f"/{model_name}/predictor/" in key:
                entry["total_predictor_size_bytes"] += size
            if key.endswith("automl_predictor_notebook.ipynb") and f"/{model_name}/notebooks/" in key:
                entry["notebook_key"] = key

    for entry in metrics_by_model.values():
        entry["total_predictor_size_mb"] = round(entry["total_predictor_size_bytes"] / (1024 * 1024), 2)

    return list(metrics_by_model.values())


def _find_top_model_predictor_prefix(s3_client, bucket: str, run_prefix: str, model_name: str) -> str | None:
    """Find the S3 prefix (with trailing slash) for a model's predictor directory.

    Scans objects under ``run_prefix`` and returns the key prefix that ends at
    ``/<model_name>/predictor/`` (inclusive of the trailing slash) so that the
    KServe storage initializer treats it unambiguously as a directory listing.
    Returns None if no matching object is found.
    """
    objects = _list_s3_objects(s3_client, bucket, run_prefix)
    needle = f"/{model_name}/predictor/"
    for obj in objects:
        key = obj["Key"]
        idx = key.find(needle)
        if idx != -1:
            return key[: idx + len(needle)]  # keep trailing slash
    return None


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


def _find_leaderboard_html(s3_client, bucket: str, run_prefix: str) -> tuple[str | None, str | None]:
    """Find the leaderboard HTML artifact produced by leaderboard_evaluation in S3.

    KFP stores the artifact under a node-scoped key that contains the artifact name
    ``html_artifact`` in the path. Returns (s3_key, html_content) on success or
    (None, None) if not found.
    """
    objects = _list_s3_objects(s3_client, bucket, run_prefix)
    for obj in objects:
        key = obj["Key"]
        if "html_artifact" in key:
            try:
                resp = s3_client.get_object(Bucket=bucket, Key=key)
                content = resp["Body"].read().decode("utf-8")
                return key, content
            except Exception as exc:
                logger.warning("Failed to read leaderboard HTML s3://%s/%s: %s", bucket, key, exc)
                return key, None
    return None, None


def _find_test_dataset_csv(s3_client, bucket: str, run_prefix: str) -> str | None:
    """Find the sampled_test_dataset artifact produced by automl_data_loader in S3.

    KFP stores the artifact under a node-scoped key that contains the artifact name
    ``sampled_test_dataset`` in the path. Returns the S3 key or None if not found.
    """
    objects = _list_s3_objects(s3_client, bucket, run_prefix)
    for obj in objects:
        if "sampled_test_dataset" in obj["Key"]:
            return obj["Key"]
    return None


# ---------------------------------------------------------------------------
# Pipeline run failure diagnostics (pod log retrieval)
# ---------------------------------------------------------------------------


def _derive_k8s_api_url(kfp_url: str | None) -> str | None:
    """Derive OpenShift API server URL from a KFP route URL.

    Standard OCP: https://<route>.apps.<cluster-domain> -> https://api.<cluster-domain>:6443
    ROSA:         https://<route>.apps.rosa.<cluster-domain> -> https://api.<cluster-domain>:443

    Override entirely with K8S_API_URL env var, or just the port with K8S_API_PORT.
    """
    override = os.environ.get("K8S_API_URL")
    if override:
        return override.strip().rstrip("/")

    if not kfp_url:
        return None

    from urllib.parse import urlparse

    hostname = urlparse(kfp_url).hostname or ""
    apps_idx = hostname.find(".apps.")
    if apps_idx < 0:
        return None
    base_domain = hostname[apps_idx + len(".apps.") :]
    is_rosa = base_domain.startswith("rosa.")
    if is_rosa:
        base_domain = base_domain[len("rosa.") :]
    default_port = 443 if is_rosa else 6443
    port = os.environ.get("K8S_API_PORT", str(default_port)).strip()
    return f"https://api.{base_domain}:{port}"


def _make_k8s_core_api(token: str, kfp_url: str | None):
    """Create a Kubernetes CoreV1Api client authenticated with a bearer token."""
    from kubernetes import client as k8s_client

    api_url = _derive_k8s_api_url(kfp_url)
    if not api_url:
        raise RuntimeError(f"Cannot derive K8S API URL from KFP URL: {kfp_url!r}")

    verify_ssl = os.environ.get("KFP_VERIFY_SSL", "true").strip().lower()
    verify_ssl = verify_ssl not in ("0", "false", "no")

    configuration = k8s_client.Configuration()
    configuration.host = api_url
    configuration.api_key = {"authorization": f"Bearer {token}"}
    configuration.verify_ssl = verify_ssl
    if not verify_ssl:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    return k8s_client.CoreV1Api(api_client=k8s_client.ApiClient(configuration))


def _is_pod_failed(pod) -> bool:
    """Return True if a pod is in a failed state."""
    phase = (pod.status.phase or "") if pod.status else ""
    if phase.lower() == "failed":
        return True
    for cs in (pod.status.container_statuses or []) if pod.status else []:
        terminated = cs.state.terminated if cs.state else None
        if terminated and terminated.exit_code != 0:
            return True
    return False


def _append_failed_pod_logs(
    run_id: str,
    namespace: str | None,
    lines: list[str],
    token: str | None = None,
    kfp_url: str | None = None,
) -> None:
    """Find failed pods for a pipeline run by label and append their logs to *lines*.

    Lists pods matching ``pipeline/runid=<run_id>`` in the given namespace,
    filters for failed pods, and fetches logs from each container.
    """
    if not token or not kfp_url:
        lines.append("\n[Missing RHOAI_TOKEN or RHOAI_KFP_URL; skipping pod log fetch]")
        return

    try:
        import kubernetes  # noqa: F401
    except ImportError:
        lines.append("\n[kubernetes package not installed; skipping pod log fetch]")
        return

    ns = namespace or "default"
    try:
        api = _make_k8s_core_api(token, kfp_url)
    except Exception as e:
        lines.append(f"\n[Could not create Kubernetes client: {e}]")
        return

    try:
        pod_list = api.list_namespaced_pod(
            namespace=ns,
            label_selector=f"pipeline/runid={run_id}",
            _request_timeout=30,
        )
    except Exception as e:
        lines.append(f"\n[Could not list pods in namespace {ns!r}: {e}]")
        return

    if not pod_list.items:
        lines.append(f"\n[No pods found with label pipeline/runid={run_id} in namespace {ns!r}]")
        return

    failed_pods = [p for p in pod_list.items if _is_pod_failed(p)]

    if not failed_pods:
        all_phases = ", ".join(f"{p.metadata.name}={p.status.phase if p.status else 'unknown'}" for p in pod_list.items)
        lines.append(f"\n[No failed pods among {len(pod_list.items)} pods: {all_phases}]")
        return

    lines.append(f"\nFound {len(failed_pods)} failed pod(s) out of {len(pod_list.items)} total")

    for pod in failed_pods:
        pod_name = pod.metadata.name
        phase = pod.status.phase if pod.status else "unknown"
        lines.append(f"\n--- Failed pod: {pod_name} (phase: {phase}) ---")

        containers = [c.name for c in (pod.spec.containers or [])] if pod.spec else []
        for container_name in containers:
            try:
                log = api.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=ns,
                    container=container_name,
                    tail_lines=100,
                    _request_timeout=60,
                )
                lines.append(f"[container: {container_name}]")
                lines.append(log if log else "(empty)")
            except Exception as e:
                lines.append(f"[container: {container_name}] error fetching logs: {e}")


def _collect_failure_details(client, run_id: str, config: dict | None = None) -> str:
    """Collect failure details from a failed pipeline run and return a formatted string.

    Fetches run-level and task-level error info from the KFP API, then retrieves
    pod logs via the Kubernetes API using the ``pipeline/runid`` label selector.

    Args:
        client: KFP client instance.
        run_id: The pipeline run ID.
        config: Integration config dict with ``rhoai_token``, ``rhoai_kfp_url``,
            and ``rhoai_project`` keys (from ``rhoai_integration_config`` fixture).

    Returns:
        Formatted multi-line string with failure details and pod logs.
    """
    lines = [f"\n{'=' * 80}", f"FAILURE DETAILS FOR RUN: {run_id}", "=" * 80]

    # --- Run-level and task-level details from KFP v2 API ---
    try:
        run_detail = client.get_run(run_id)
        run_obj = getattr(run_detail, "run", run_detail)

        run_error = getattr(run_obj, "error", None)
        if run_error:
            error_msg = getattr(run_error, "message", str(run_error))
            lines.append(f"\nRUN ERROR: {error_msg}")

        rd = getattr(run_obj, "run_details", None)
        task_list = getattr(rd, "task_details", None) if rd else None

        if task_list:
            _INTERNAL_SUFFIXES = ("-driver",)
            _INTERNAL_NAMES = ("root", "executor")

            for task in task_list:
                name = getattr(task, "display_name", None) or getattr(task, "task_id", "?")
                state = getattr(task, "state", None)
                state_str = str(state).upper() if state else "NOT_STARTED"

                if name in _INTERNAL_NAMES or any(name.endswith(s) for s in _INTERNAL_SUFFIXES):
                    continue

                if state_str in ("FAILED", "ERROR", "SYSTEM_ERROR"):
                    lines.append(f"\nFAILED TASK: {name}")
                    lines.append(f"  State: {state_str}")
                    task_error = getattr(task, "error", None)
                    if task_error:
                        error_msg = getattr(task_error, "message", str(task_error))
                        lines.append(f"  Error: {error_msg}")
                    start = getattr(task, "start_time", None)
                    end = getattr(task, "end_time", None)
                    if start and end:
                        lines.append(f"  Duration: {start} -> {end}")
                else:
                    lines.append(f"  TASK: {name} — {state_str}")
        else:
            lines.append("\n[No task_details in run response]")
    except Exception as e:
        lines.append(f"\n[Could not fetch run details from KFP API: {e}]")

    # --- Pod logs via Kubernetes API (label-based pod discovery) ---
    try:
        namespace = config.get("rhoai_project") if config else None
        token = config.get("rhoai_token") if config else None
        kfp_url = config.get("rhoai_kfp_url") if config else None
        _append_failed_pod_logs(run_id, namespace, lines, token=token, kfp_url=kfp_url)
    except Exception as e:
        lines.append(f"\n[Could not fetch pod logs: {e}]")

    lines.append("=" * 80)
    return "\n".join(lines)


def _fetch_pod_logs_str(v1, namespace: str, label_selector: str, tail_lines: int = 100) -> str:
    """Fetch logs from all pods matching *label_selector* and return a formatted string.

    Unlike ``_append_failed_pod_logs`` (which filters for failed pods only), this
    function fetches logs from every pod matching the selector — useful for Job pods
    and KServe predictor pods that may still be running or in CrashLoopBackOff.
    """
    lines = []
    try:
        pod_list = v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=label_selector,
            _request_timeout=30,
        )
        if not pod_list.items:
            return f"[No pods found with label selector {label_selector!r} in namespace {namespace!r}]"

        lines.append(f"Pod logs for label selector {label_selector!r} ({len(pod_list.items)} pod(s)):")
        for pod in pod_list.items:
            pod_name = pod.metadata.name
            phase = pod.status.phase if pod.status else "unknown"
            lines.append(f"\n--- Pod: {pod_name} (phase: {phase}) ---")
            containers = [c.name for c in (pod.spec.containers or [])] if pod.spec else []
            for container_name in containers:
                try:
                    log = v1.read_namespaced_pod_log(
                        name=pod_name,
                        namespace=namespace,
                        container=container_name,
                        tail_lines=tail_lines,
                        _request_timeout=60,
                    )
                    lines.append(f"[container: {container_name}]")
                    lines.append(log if log else "(empty)")
                except Exception as e:
                    lines.append(f"[container: {container_name}] error fetching logs: {e}")
    except Exception as e:
        return f"[Could not fetch pod logs for {label_selector!r}: {e}]"
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# KServe deployment helpers
# ---------------------------------------------------------------------------


def _make_notebook_job_name(run_id: str, model_name: str) -> str:
    """Return a valid Kubernetes Job name (≤63 chars, DNS label safe).

    Layout: ``nb-`` (3) + clean model name (≤46) + ``-`` (1) + run_id[:8] (8) = ≤58 chars.
    """
    clean = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")[:46]
    return f"nb-{clean}-{run_id[:8]}"


def _make_isvc_name(scenario_id: str, run_id: str) -> str:
    """Return a valid Kubernetes name for an InferenceService (≤36 chars, DNS label safe).

    The name is capped at 36 characters because odh-model-controller generates a
    kube-rbac-proxy ConfigMap/volume named ``{isvc_name}-kube-rbac-proxy-sar-config``
    (27-char suffix).  Kubernetes volume names must be ≤ 63 characters, so:
        36 (isvc_name) + 27 (suffix) = 63  ← exactly at the limit.

    Layout: ``automl-`` (7) + clean (≤20) + ``-`` (1) + run_id[:8] (8) = ≤ 36.
    """
    clean = re.sub(r"[^a-z0-9]+", "-", scenario_id.lower()).strip("-")[:20]
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


def _create_kserve_s3_secret(v1, namespace: str, secret_name: str, bucket: str, integration_config: dict) -> None:
    """Create (or replace) an RHOAI Data Connection secret for KServe storage initializer.

    The secret includes the ``opendatahub.io/managed: "true"`` label so that
    odh-model-controller recognises it as a Data Connection and wires it up as the
    storage key for the InferenceService predictor.
    """
    from kubernetes import client
    from kubernetes.client.rest import ApiException

    endpoint = integration_config["s3_endpoint"]

    # Use the exact secret data keys and metadata that the RHOAI Dashboard writes
    # when creating an S3 Data Connection (matched against a known working secret).
    # Keys: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT, AWS_S3_BUCKET.
    # Labels: opendatahub.io/managed + opendatahub.io/dashboard (no connection-type label).
    # Annotations: connection-type, connection-type-protocol, connection-type-ref,
    #              and openshift.io/display-name.
    string_data: dict[str, str] = {
        "AWS_ACCESS_KEY_ID": integration_config["s3_access_key"],
        "AWS_SECRET_ACCESS_KEY": integration_config["s3_secret_key"],
        "AWS_S3_ENDPOINT": endpoint,
        "AWS_S3_BUCKET": bucket,
    }

    secret = client.V1Secret(
        metadata=client.V1ObjectMeta(
            name=secret_name,
            namespace=namespace,
            labels={
                "opendatahub.io/managed": "true",
                "opendatahub.io/dashboard": "true",
            },
            annotations={
                "opendatahub.io/connection-type": "s3",
                "opendatahub.io/connection-type-protocol": "s3",
                "opendatahub.io/connection-type-ref": "s3",
                "openshift.io/display-name": secret_name,
            },
        ),
        type="Opaque",
        string_data=string_data,
    )
    try:
        v1.create_namespaced_secret(namespace, secret, _request_timeout=_K8S_CALL_TIMEOUT)
    except ApiException as e:
        if e.status == 409:
            v1.replace_namespaced_secret(secret_name, namespace, secret, _request_timeout=_K8S_CALL_TIMEOUT)
        else:
            raise


def _ensure_serving_runtime(co, namespace: str, runtime_name: str, serving_image: str) -> bool:
    """Create the AutoGluon ServingRuntime if it does not exist in the namespace.

    Returns:
        True if the ServingRuntime was newly created, False if it already existed.
    """
    from kubernetes.client.rest import ApiException

    try:
        co.get_namespaced_custom_object(
            group=_KSERVE_GROUP,
            version=_KSERVE_SR_VERSION,
            namespace=namespace,
            plural=_KSERVE_SR_PLURAL,
            name=runtime_name,
            _request_timeout=_K8S_CALL_TIMEOUT,
        )
        logger.info("ServingRuntime %r already exists in %r — skipping creation", runtime_name, namespace)
        return False
    except ApiException as e:
        if e.status != 404:
            raise

    sr_annotations = {
        "opendatahub.io/apiProtocol": "REST",
        "opendatahub.io/serving-runtime-scope": "global",
        "opendatahub.io/template-display-name": "AutoGluon ServingRuntime for KServe",
        "openshift.io/display-name": "AutoGluon ServingRuntime for KServe",
    }

    runtime = {
        "apiVersion": f"{_KSERVE_GROUP}/{_KSERVE_SR_VERSION}",
        "kind": "ServingRuntime",
        "metadata": {
            "name": runtime_name,
            "namespace": namespace,
            "annotations": sr_annotations,
        },
        "spec": {
            "annotations": {
                "prometheus.kserve.io/path": "/metrics",
                "prometheus.kserve.io/port": "8080",
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
        _request_timeout=_K8S_CALL_TIMEOUT,
    )
    logger.info("Created ServingRuntime %r in %r", runtime_name, namespace)
    return True


def _create_connection_sa(v1, namespace: str, secret_name: str) -> str:
    """Create the companion ServiceAccount required by odh-model-controller for a Data Connection.

    When the RHOAI Dashboard registers a Data Connection it creates a ServiceAccount
    named ``{secret_name}-sa`` alongside the secret.  odh-model-controller checks for
    this SA before setting ``storage.key`` / ``serviceAccountName`` on an ISVC.
    For temporary test secrets we must create the SA ourselves so the controller
    can proceed with its normal reconcile loop.

    Returns:
        The ServiceAccount name (``{secret_name}-sa``).
    """
    from kubernetes import client as k8s_client
    from kubernetes.client.rest import ApiException

    sa_name = f"{secret_name}-sa"
    sa = k8s_client.V1ServiceAccount(
        metadata=k8s_client.V1ObjectMeta(
            name=sa_name,
            namespace=namespace,
            # No opendatahub.io labels: Dashboard-created companion SAs have none
            # (confirmed from cluster inspection of real working deployments).
            # The controller looks up the SA by name, not via label-filtered informer.
        ),
        # List the Data Connection secret in the SA's secrets field, exactly as
        # the RHOAI Dashboard does when registering a Data Connection.
        secrets=[k8s_client.V1ObjectReference(name=secret_name)],
    )
    try:
        v1.create_namespaced_service_account(namespace, sa, _request_timeout=_K8S_CALL_TIMEOUT)
        logger.info("Created ServiceAccount %r in namespace %r (secrets=[%r])", sa_name, namespace, secret_name)
    except ApiException as exc:
        if exc.status == 409:
            logger.info("ServiceAccount %r already exists — reusing", sa_name)
        else:
            raise
    return sa_name


def _create_connection_rbac(rbac_v1, namespace: str, sa_name: str, secret_name: str) -> str:
    """Create a Role + RoleBinding so the SA can GET the Data Connection secret.

    The KServe agent sidecar authenticates to the Kubernetes API using the pod's
    mounted SA token to read the credentials secret.  Without this RBAC the agent
    receives a 403 Forbidden, skips downloading, and the kserve-container crashes
    with 'predictor.pkl not found'.

    The RHOAI Dashboard creates equivalent RBAC when registering a Data Connection;
    we replicate it here for temporary test secrets.

    Returns:
        The RoleBinding name (same as ``sa_name`` for simplicity).
    """
    from kubernetes import client as k8s_client
    from kubernetes.client.rest import ApiException

    role_name = sa_name
    role = k8s_client.V1Role(
        metadata=k8s_client.V1ObjectMeta(
            name=role_name,
            namespace=namespace,
            labels={
                "opendatahub.io/managed": "true",
                "opendatahub.io/dashboard": "true",
            },
        ),
        rules=[
            k8s_client.V1PolicyRule(
                api_groups=[""],
                resources=["secrets"],
                verbs=["get"],
                resource_names=[secret_name],
            )
        ],
    )
    try:
        rbac_v1.create_namespaced_role(namespace, role, _request_timeout=_K8S_CALL_TIMEOUT)
        logger.info("Created Role %r in namespace %r", role_name, namespace)
    except ApiException as exc:
        if exc.status == 409:
            logger.info("Role %r already exists — reusing", role_name)
        else:
            raise

    rb_name = sa_name
    role_binding = k8s_client.V1RoleBinding(
        metadata=k8s_client.V1ObjectMeta(
            name=rb_name,
            namespace=namespace,
            labels={
                "opendatahub.io/managed": "true",
                "opendatahub.io/dashboard": "true",
            },
        ),
        subjects=[
            k8s_client.RbacV1Subject(
                kind="ServiceAccount",
                name=sa_name,
                namespace=namespace,
            )
        ],
        role_ref=k8s_client.V1RoleRef(
            api_group="rbac.authorization.k8s.io",
            kind="Role",
            name=role_name,
        ),
    )
    try:
        rbac_v1.create_namespaced_role_binding(namespace, role_binding, _request_timeout=_K8S_CALL_TIMEOUT)
        logger.info("Created RoleBinding %r in namespace %r", rb_name, namespace)
    except ApiException as exc:
        if exc.status == 409:
            logger.info("RoleBinding %r already exists — reusing", rb_name)
        else:
            raise

    return rb_name


def _ensure_deployment_storage_annotations(
    apps_v1,
    namespace: str,
    isvc_name: str,
    storage_key: str,
    artifacts_bucket: str,
    storage_path: str,
    wait_seconds: int = 60,
) -> bool:
    """Wait for the predictor Deployment and ensure it has storage initializer annotations.

    odh-model-controller sets ``internal.serving.kserve.io/storage-initializer-sourceuri``
    and ``storage-spec-key`` on the predictor Deployment (and its pod template) when it
    processes the ``opendatahub.io/connections`` annotation on the ISVC.  Without these
    annotations the kserve-container never downloads the model and crashes immediately.

    If the controller fails to set the annotations (connection processing silently skipped),
    this function patches the Deployment and pod template directly so the pods get the
    storage spec on their next restart.

    Args:
        apps_v1: Kubernetes AppsV1Api client.
        namespace: Target namespace.
        isvc_name: InferenceService (and Deployment name prefix).
        storage_key: Name of the RHOAI Data Connection secret.
        artifacts_bucket: S3 bucket name containing the model artifacts.
        storage_path: S3 path (relative to bucket root) ending at the predictor directory.
        wait_seconds: Seconds to poll for the Deployment to appear (default: 60).

    Returns:
        True if the Deployment was found (with or without patching), False if it never appeared.
    """
    from kubernetes.client.rest import ApiException

    deployment_name = f"{isvc_name}-predictor"
    deadline = time.monotonic() + wait_seconds

    dep = None
    while time.monotonic() < deadline:
        try:
            dep = apps_v1.read_namespaced_deployment(deployment_name, namespace, _request_timeout=_K8S_CALL_TIMEOUT)
            break
        except ApiException as exc:
            if exc.status == 404:
                remaining = deadline - time.monotonic()
                logger.info(
                    "Deployment %r not yet created — waiting (%.0fs remaining)...",
                    deployment_name,
                    max(0, remaining),
                )
                time.sleep(5)
            else:
                logger.warning("Could not read Deployment %r: HTTP %s", deployment_name, exc.status)
                break
        except Exception as exc:
            logger.warning("Could not read Deployment %r: %s", deployment_name, exc)
            break

    if dep is None:
        logger.warning("Deployment %r did not appear within %ds", deployment_name, wait_seconds)
        return False

    ann = dep.metadata.annotations or {}
    spec_key = ann.get("internal.serving.kserve.io/storage-spec-key", "")
    source_uri = ann.get("internal.serving.kserve.io/storage-initializer-sourceuri", "")

    logger.info(
        "Deployment %r annotations — storage-spec-key=%r  storage-initializer-sourceuri=%r  storage-spec=%r",
        deployment_name,
        spec_key or "(not set)",
        source_uri or "(not set)",
        ann.get("internal.serving.kserve.io/storage-spec", "(not set)"),
    )

    if spec_key and source_uri:
        logger.info(
            "Deployment %r already has storage annotations — odh-model-controller processed the connection",
            deployment_name,
        )
        return True

    # odh-model-controller did not set the storage annotations (connection processing
    # failed silently — no K8s event is emitted for this case).
    # Patch the Deployment and pod template directly so new pods get the storage spec.
    # The storage-initializer-sourceuri uses the S3 URI format: s3://{bucket}/{path}.
    # The kserve-container reads storage-spec-key to look up S3 credentials and
    # storage-initializer-sourceuri to locate the model files.
    storage_uri = f"s3://{artifacts_bucket}/{storage_path.rstrip('/')}"
    storage_annotations = {
        "internal.serving.kserve.io/storage-initializer-sourceuri": storage_uri,
        "internal.serving.kserve.io/storage-spec-key": storage_key,
        "internal.serving.kserve.io/storage-spec": "true",
    }
    patch_body = {
        "metadata": {"annotations": storage_annotations},
        "spec": {"template": {"metadata": {"annotations": storage_annotations}}},
    }
    try:
        apps_v1.patch_namespaced_deployment(deployment_name, namespace, patch_body, _request_timeout=_K8S_CALL_TIMEOUT)
        logger.info(
            "Patched Deployment %r with storage annotations (uri=%r, key=%r) — "
            "pods will restart and download the model",
            deployment_name,
            storage_uri,
            storage_key,
        )
    except Exception as exc:
        logger.warning(
            "Failed to patch Deployment %r with storage annotations: %s — model download may fail",
            deployment_name,
            exc,
        )

    return True


def _log_isvc_events(v1, namespace: str, isvc_name: str) -> None:
    """Read and log Kubernetes Events for the InferenceService.

    Controllers emit Warning events when they fail to process an ISVC (e.g.
    secret not found, invalid connection format, SA missing).  These events
    surface the exact error reason so we don't have to guess.
    """
    try:
        events = v1.list_namespaced_event(
            namespace,
            field_selector=f"involvedObject.name={isvc_name},involvedObject.kind=InferenceService",
            _request_timeout=_K8S_CALL_TIMEOUT,
        )
        if not events.items:
            logger.info("ISVC %r: no Kubernetes events found", isvc_name)
            return
        for evt in events.items:
            logger.info(
                "ISVC %r event: type=%s reason=%r message=%r count=%s source=%s",
                isvc_name,
                evt.type,
                evt.reason or "",
                evt.message or "",
                evt.count,
                getattr(evt.source, "component", ""),
            )
    except Exception as exc:
        logger.warning("Could not read events for ISVC %r: %s", isvc_name, exc)


def _create_inference_service(
    co,
    namespace: str,
    isvc_name: str,
    runtime_name: str,
    storage_path: str,
    storage_key: str,
    hardware_profile_name: str = "default-profile",
    hardware_profile_namespace: str = "redhat-ods-applications",
    hardware_profile_resource_version: str = "",
) -> None:
    """Create a KServe InferenceService in RawDeployment mode with an external Route.

    Call this after the Data Connection secret and its companion ServiceAccount
    have been created and the informer settle wait has elapsed, so that
    odh-model-controller can set ``storage.key`` / ``serviceAccountName`` on the
    ISVC in the same reconcile loop that creates the predictor Deployment.

    Key annotations:
    - ``serving.kserve.io/stop: 'false'`` — REQUIRED. Without this the controller
      skips Deployment creation (only Service + Route are created).
    - ``serving.kserve.io/deploymentMode: RawDeployment`` — plain K8s Deployment
      instead of Serverless (Knative).
    - ``networking.kserve.io/visibility: exposed`` (label) — creates an external Route.
    - ``security.opendatahub.io/enable-auth: 'true'`` — kube-rbac-proxy sidecar.
    - ``opendatahub.io/connections`` / ``opendatahub.io/connection-path`` — the
      controller reads these to set storage.key / storage.path / serviceAccountName.
    - ``opendatahub.io/hardware-profile-*`` — required by odh-model-controller to
      look up the hardware profile before creating the Deployment.
    """
    from kubernetes.client.rest import ApiException

    annotations = {
        # REQUIRED: without 'false' the controller skips Deployment creation.
        "serving.kserve.io/stop": "false",
        # Required: tells the controller to use RawDeployment (plain K8s Deployment)
        # instead of Serverless (Knative). Without this the ISVC stays in Pending
        # state indefinitely when Knative is not configured on the cluster.
        # Note: this is set via POST Create (not SSA), matching how the RHOAI
        # dashboard sets it — that is why it does not appear in the working ISVC's
        # SSA managedFields but is present in the actual ISVC annotations.
        "serving.kserve.io/deploymentMode": "RawDeployment",
        "security.opendatahub.io/enable-auth": "true",
        "openshift.io/display-name": isvc_name,
        "openshift.io/description": "",
        # Controller reads these to create the ServiceAccount ({storage_key}-sa)
        # and confirm storage settings. storage.key / storage.path are also set
        # directly in the spec (below) so the agent does not depend on controller
        # annotation processing completing before the pod starts.
        # Strip trailing slash to match the format used by the RHOAI Dashboard —
        # the working isvc.yaml shows the annotation without a trailing slash.
        "opendatahub.io/connections": storage_key,
        "opendatahub.io/connection-path": storage_path.rstrip("/"),
        "opendatahub.io/model-type": "predictive",
        # Hardware profile — controller looks this up before creating the Deployment.
        "opendatahub.io/hardware-profile-name": hardware_profile_name,
        "opendatahub.io/hardware-profile-namespace": hardware_profile_namespace,
    }
    if hardware_profile_resource_version:
        annotations["opendatahub.io/hardware-profile-resource-version"] = hardware_profile_resource_version

    isvc = {
        "apiVersion": f"{_KSERVE_GROUP}/{_KSERVE_ISVC_VERSION}",
        "kind": "InferenceService",
        "metadata": {
            "name": isvc_name,
            "namespace": namespace,
            "labels": {
                "networking.kserve.io/visibility": "exposed",
                "opendatahub.io/dashboard": "true",
            },
            "annotations": annotations,
        },
        "spec": {
            "predictor": {
                # Matches the working ISVC: service account token is not auto-mounted
                # because the agent uses the Data Connection secret for S3 access.
                "automountServiceAccountToken": False,
                # SA created alongside the secret so odh-model-controller will wire
                # up storage.key / serviceAccountName in its reconcile loop.
                "serviceAccountName": f"{storage_key}-sa",
                "deploymentStrategy": {"type": "RollingUpdate"},
                "maxReplicas": 1,
                "minReplicas": 1,
                "model": {
                    "modelFormat": {"name": "autogluon", "version": "1"},
                    # Empty string matches the working ISVC spec.
                    "name": "",
                    "runtime": runtime_name,
                    "resources": {
                        "requests": {"cpu": "2", "memory": "4Gi"},
                        "limits": {"cpu": "2", "memory": "4Gi"},
                    },
                    # Set storage directly so the agent has the URI from pod start,
                    # matching the working isvc.yaml spec. The controller also sets
                    # these via annotation reconciliation — setting them here ensures
                    # the agent never starts without a storage spec.
                    "storage": {
                        "key": storage_key,
                        "path": storage_path.rstrip("/"),
                    },
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
            _request_timeout=_K8S_CALL_TIMEOUT,
        )
    except ApiException as e:
        if e.status != 409:
            raise


def _wait_for_isvc_ready(
    co,
    namespace: str,
    isvc_name: str,
    timeout_seconds: int = 300,
    poll_interval: int = 30,
) -> tuple[bool, str | None]:
    """Poll an InferenceService until Ready=True, a terminal failure appears, or timeout.

    At each poll the following are logged to surface controller issues early:
    - ``spec.predictor.model.storage.key`` and ``spec.predictor.serviceAccountName`` —
      set by odh-model-controller; absent means the controller has not yet started
      processing the ISVC.
    - All status conditions with reason/message — pinpoints blocking errors such as
      ``ServingRuntimeNotFound`` or ``InvalidStorageSpec``.

    Args:
        co: Kubernetes CustomObjectsApi client.
        namespace: Target namespace.
        isvc_name: InferenceService name to watch.
        timeout_seconds: Maximum seconds to wait before returning (default: 300).
        poll_interval: Seconds between status polls (default: 30).

    Returns:
        ``(is_ready, blocking_reason)`` where:
        - ``is_ready`` is ``True`` only when the ISVC status condition ``Ready=True``.
        - ``blocking_reason`` is a non-``None`` description string when a terminal
          failure condition is detected; the caller should abort scoring immediately.
    """
    from kubernetes.client.rest import ApiException

    # Reasons that indicate the controller cannot recover without human intervention.
    _BLOCKING_REASONS = frozenset(
        {
            "ServingRuntimeNotFound",
            "NoSupportedRuntime",
            "InvalidStorageSpec",
            "RuntimeNotRecognized",
            "UnsupportedProtocol",
        }
    )

    start = time.monotonic()
    last_cond_fingerprint: frozenset = frozenset()

    while True:
        elapsed = time.monotonic() - start
        try:
            isvc = co.get_namespaced_custom_object(
                group=_KSERVE_GROUP,
                version=_KSERVE_ISVC_VERSION,
                namespace=namespace,
                plural=_KSERVE_ISVC_PLURAL,
                name=isvc_name,
                _request_timeout=_K8S_CALL_TIMEOUT,
            )
        except ApiException as exc:
            logger.warning("ISVC %r: GET failed (elapsed %.0fs, HTTP %s)", isvc_name, elapsed, exc.status)
        except Exception as exc:
            logger.warning("ISVC %r: GET failed (elapsed %.0fs): %s", isvc_name, elapsed, exc)
        else:
            status = isvc.get("status") or {}
            spec = isvc.get("spec") or {}
            conditions = status.get("conditions") or []

            # Log controller-populated spec fields at every poll.
            # These are set by odh-model-controller once it processes the ISVC.
            # If they remain "(not set)" the controller has not started processing yet.
            predictor = spec.get("predictor") or {}
            model = predictor.get("model") or {}
            storage = model.get("storage") or {}
            logger.info(
                "ISVC %r (elapsed %.0fs): storage.key=%r  serviceAccountName=%r  servingRuntime=%r  deploymentMode=%r",
                isvc_name,
                elapsed,
                storage.get("key") or "(not set)",
                predictor.get("serviceAccountName") or "(not set)",
                status.get("servingRuntimeName", ""),
                status.get("deploymentMode", ""),
            )

            # Log conditions only when they change (avoids log spam on repeated polls).
            cond_fingerprint = frozenset((c.get("type"), c.get("status"), c.get("reason", "")) for c in conditions)
            if cond_fingerprint != last_cond_fingerprint:
                if conditions:
                    for cond in conditions:
                        ctype = cond.get("type", "?")
                        cstatus = cond.get("status", "?")
                        reason = cond.get("reason", "")
                        message = cond.get("message", "")
                        detail = f" | reason={reason}" if reason else ""
                        detail += f" | message={message!r}" if message else ""
                        logger.info("ISVC %r condition %s=%s%s", isvc_name, ctype, cstatus, detail)
                else:
                    logger.info(
                        "ISVC %r: no status conditions yet (elapsed %.0fs)",
                        isvc_name,
                        elapsed,
                    )
                last_cond_fingerprint = cond_fingerprint

            # Detect terminal (non-recoverable) failures early to avoid waiting
            # the full timeout.
            for cond in conditions:
                if cond.get("status") == "False":
                    reason = cond.get("reason", "")
                    if reason in _BLOCKING_REASONS:
                        blocking = f"{cond.get('type')}=False reason={reason}: {cond.get('message', '')}"
                        logger.error(
                            "ISVC %r: terminal failure after %.0fs — %s",
                            isvc_name,
                            elapsed,
                            blocking,
                        )
                        return False, blocking

            # Check for the ISVC-level Ready condition.
            cond_map = {c.get("type"): c.get("status") for c in conditions}
            if cond_map.get("Ready") == "True":
                logger.info("ISVC %r: Ready=True after %.0fs", isvc_name, elapsed)
                return True, None

        if elapsed >= timeout_seconds:
            logger.warning("ISVC %r: timed out after %.0fs without Ready=True", isvc_name, elapsed)
            return False, None

        sleep_secs = min(poll_interval, timeout_seconds - elapsed)
        logger.info(
            "ISVC %r: not yet Ready — next poll in %.0fs (%.0fs remaining)...",
            isvc_name,
            sleep_secs,
            timeout_seconds - elapsed,
        )
        time.sleep(sleep_secs)


def _resolve_isvc_external_url(co, namespace: str, isvc_name: str) -> str | None:
    """Return the external HTTPS URL for an InferenceService (single attempt, no polling).

    Primary: reads ``status.url`` from the ISVC itself — KServe sets this to the
    external Route URL as soon as IngressReady is True (typically within seconds).
    The internal ``svc.cluster.local`` address is filtered out.

    Fallback: scans OpenShift Routes for ``{isvc_name}-predictor`` or ``{isvc_name}``.

    Returns ``https://...`` on success, or ``None`` if not yet available.
    """
    from kubernetes.client.rest import ApiException

    # Primary: ISVC status.url (set by KServe when ingress is ready).
    try:
        isvc = co.get_namespaced_custom_object(
            group=_KSERVE_GROUP,
            version=_KSERVE_ISVC_VERSION,
            namespace=namespace,
            plural=_KSERVE_ISVC_PLURAL,
            name=isvc_name,
            _request_timeout=_K8S_CALL_TIMEOUT,
        )
        status_url = (isvc.get("status") or {}).get("url", "")
        if status_url.startswith("https://") and ".svc.cluster.local" not in status_url:
            logger.info("Resolved external URL from ISVC status.url: %s", status_url)
            return status_url
    except ApiException:
        pass

    # Fallback: Route lookup — KServe RawDeployment names the Route after the ISVC
    # ({isvc_name}), while the Service is named {isvc_name}-predictor.
    def _extract_host(route: dict) -> str | None:
        host = (route.get("spec") or {}).get("host")
        if not host:
            ingress = (route.get("status") or {}).get("ingress") or []
            host = ingress[0].get("host") if ingress else None
        return host

    for route_name in (isvc_name, f"{isvc_name}-predictor"):
        try:
            route = co.get_namespaced_custom_object(
                group="route.openshift.io",
                version="v1",
                namespace=namespace,
                plural="routes",
                name=route_name,
                _request_timeout=_K8S_CALL_TIMEOUT,
            )
            host = _extract_host(route)
            if host:
                logger.info("Resolved external URL from Route %r: https://%s", route_name, host)
                return f"https://{host}"
        except ApiException as exc:
            if exc.status != 404:
                logger.debug("Route lookup %r: HTTP %s", route_name, exc.status)

    return None


def _column_sample_to_instances(sample: list[dict]) -> list[dict]:
    """Convert column-oriented [{col: [val, ...]}] to per-row instance dicts with list values.

    The functional test config stores samples in column (pandas) format for readability.
    The AutoGluon KServe server's ``get_predict_input`` builds each instance into a
    ``pd.DataFrame`` via ``pd.DataFrame(instance, columns=columns)``.  pandas requires
    each column value to be a sequence (not a scalar), so each row is emitted as
    ``{col: [val], ...}`` rather than ``{col: val, ...}``.
    """
    if not sample:
        return []
    col_data = sample[0]  # {col: [val, ...]}
    n_rows = len(next(iter(col_data.values()), []))
    return [{col: [values[i]] for col, values in col_data.items()} for i in range(n_rows)]


def _score_inference_service(
    isvc_url: str,
    model_name: str,
    instances: list[dict],
    token: str | None,
    max_retries: int = 5,
    retry_interval_seconds: int = 30,
) -> dict:
    """Send a KServe v1 predict request with retry on 5xx transient errors.

    Uses a fixed retry interval for 500/502/503/504 (model still loading / pod
    starting). Exponential backoff is intentionally avoided here because the
    expected wait is model-startup time (image pull + storage initializer S3
    download + AutoGluon model load), which can take several minutes but is
    bounded — constant polling is more efficient than backing off past that window.

    Default: 20 attempts × 30 s interval = up to ~10 minutes of startup wait.

    Args:
        isvc_url: Base URL of the InferenceService (e.g. https://host).
        model_name: ISVC name used in the URL path.
        instances: Row-oriented list of instance dicts.
        token: Bearer token for RHOAI auth (may be None).
        max_retries: Total number of attempts (including the first).
        retry_interval_seconds: Fixed wait between retries on 5xx responses.

    Returns:
        Parsed JSON response dict.

    Raises:
        RuntimeError: After all retries are exhausted.
    """
    predict_url = f"{isvc_url.rstrip('/')}/v1/models/{model_name}:predict"
    payload = json.dumps({"instances": instances}).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Disable TLS verification for test clusters with self-signed certificates.
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    last_error: str = ""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(predict_url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.reason}"
            if exc.code in (500, 502, 503, 504) and attempt < max_retries - 1:
                logger.warning(
                    "Scoring attempt %d/%d got %s — model still loading; retrying in %ds",
                    attempt + 1,
                    max_retries,
                    last_error,
                    retry_interval_seconds,
                )
                time.sleep(retry_interval_seconds)
                continue
            raise RuntimeError(last_error) from exc
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries - 1:
                logger.warning(
                    "Scoring attempt %d/%d failed: %s; retrying in %ds",
                    attempt + 1,
                    max_retries,
                    last_error,
                    retry_interval_seconds,
                )
                time.sleep(retry_interval_seconds)
                continue
            raise RuntimeError(last_error) from exc

    raise RuntimeError(f"All {max_retries} scoring attempts failed. Last error: {last_error}")


def _list_hardware_profile_names(co, namespace: str) -> list[str]:
    """Best-effort list of HardwareProfile names in a namespace (for error messages)."""
    try:
        lst = co.list_namespaced_custom_object(
            group="infrastructure.opendatahub.io",
            version="v1alpha1",
            namespace=namespace,
            plural="hardwareprofiles",
            _request_timeout=_K8S_CALL_TIMEOUT,
        )
        items = lst.get("items") or []
        return sorted((i.get("metadata") or {}).get("name", "") for i in items if (i.get("metadata") or {}).get("name"))
    except Exception as exc:
        logger.warning("Could not list HardwareProfiles in %r: %s", namespace, exc)
        return []


def _fetch_hardware_profile_resource_version(co, namespace: str, name: str) -> str:
    """Fetch ``metadata.resourceVersion`` for a HardwareProfile CR with retries.

    odh-model-controller requires the ``opendatahub.io/hardware-profile-resource-version``
    InferenceService annotation to match the live HardwareProfile resourceVersion before
    it creates the predictor Deployment. If absent or stale the controller skips
    Deployment creation while Route/Service may still appear.

    Retries mitigate transient RBAC or API delays. On failure, logs HardwareProfile
    names visible in the namespace (if list is permitted).

    Returns:
        Non-empty resourceVersion string, or "" if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(_HW_PROFILE_FETCH_ATTEMPTS):
        try:
            obj = co.get_namespaced_custom_object(
                group="infrastructure.opendatahub.io",
                version="v1alpha1",
                namespace=namespace,
                plural="hardwareprofiles",
                name=name,
                _request_timeout=_K8S_CALL_TIMEOUT,
            )
            rv = (obj.get("metadata") or {}).get("resourceVersion", "")
            if rv:
                logger.info(
                    "HardwareProfile %r/%r resourceVersion=%s (attempt %d)",
                    namespace,
                    name,
                    rv,
                    attempt + 1,
                )
                return rv
            logger.warning(
                "HardwareProfile %r/%r returned empty resourceVersion (attempt %d)",
                namespace,
                name,
                attempt + 1,
            )
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "HardwareProfile GET %r/%r attempt %d/%d failed: %s",
                namespace,
                name,
                attempt + 1,
                _HW_PROFILE_FETCH_ATTEMPTS,
                exc,
            )
        if attempt < _HW_PROFILE_FETCH_ATTEMPTS - 1:
            time.sleep(_HW_PROFILE_FETCH_DELAY_SECONDS)

    available = _list_hardware_profile_names(co, namespace)
    logger.error(
        "Could not fetch resourceVersion for HardwareProfile %r in namespace %r after %d attempts. "
        "HardwareProfiles visible in namespace (if list allowed): %s. Last GET error: %s",
        name,
        namespace,
        _HW_PROFILE_FETCH_ATTEMPTS,
        available or "(none or not listable)",
        last_exc,
    )
    return ""


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
            _request_timeout=_K8S_CALL_TIMEOUT,
        )
    except ApiException as e:
        if e.status != 404:
            logger.warning("Failed to delete InferenceService %r: %s", isvc_name, e)


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
        if not succeeded:
            failure_details = _collect_failure_details(kfp_client, run_id, config=config)
            pytest.fail(
                f"Pipeline run {run_id} did not succeed for scenario {func_config.id}; "
                f"state={getattr(getattr(detail, 'run', detail), 'state', 'unknown')}" + failure_details
            )

        # ------------------------------------------------------------------
        # 5. Read metrics, model sizes, leaderboard, and test dataset from S3
        # ------------------------------------------------------------------
        run_prefix = f"{PIPELINE_DISPLAY_NAME}/{run_id}"
        s3_cleanup_tracker.track_artifact_prefix(artifacts_bucket, run_prefix)

        metrics_list = _collect_model_metrics_and_sizes(s3_client, artifacts_bucket, run_prefix)

        logger.info(
            "Scenario %s: found %d models (top_n=%d)",
            func_config.id,
            len(metrics_list),
            func_config.top_n,
        )
        for m in metrics_list:
            logger.info(
                "  Model: %s | Predictor: %.2f MB | Notebook: %s | Metrics: %s",
                m["model_name"],
                m["total_predictor_size_mb"],
                m["notebook_key"] or "(not found)",
                json.dumps(m["metrics"], default=str),
            )

        leaderboard_key, leaderboard_html = _find_leaderboard_html(s3_client, artifacts_bucket, run_prefix)
        logger.info(
            "Scenario %s: leaderboard HTML artifact — %s",
            func_config.id,
            f"s3://{artifacts_bucket}/{leaderboard_key}" if leaderboard_key else "(not found)",
        )

        test_dataset_key = _find_test_dataset_csv(s3_client, artifacts_bucket, run_prefix)
        logger.info(
            "Scenario %s: sampled_test_dataset artifact — %s",
            func_config.id,
            f"s3://{artifacts_bucket}/{test_dataset_key}" if test_dataset_key else "(not found)",
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
                run_id=run_id,
                rhoai_integration_config=config,
                temp_kubeconfig_path=temp_kubeconfig_path,
            )

        # ------------------------------------------------------------------
        # 6b. [Optional] Execute top-1 model notebook as a Kubernetes Job
        # ------------------------------------------------------------------
        notebook_run_result: dict = {}
        if os.environ.get("RHOAI_NOTEBOOK_RUNNER_IMAGE", "").strip() and metrics_list:
            notebook_run_result = self._run_notebook_test(
                notebook_key=metrics_list[0].get("notebook_key"),
                run_id=run_id,
                model_name=metrics_list[0]["model_name"],
                s3_client=s3_client,
                artifacts_bucket=artifacts_bucket,
                rhoai_integration_config=config,
                temp_kubeconfig_path=temp_kubeconfig_path,
            )
        else:
            logger.info(
                "Scenario %s: notebook execution step skipped (set RHOAI_NOTEBOOK_RUNNER_IMAGE to enable)",
                func_config.id,
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
                        "notebook_key": m["notebook_key"],
                    }
                    for m in metrics_list
                ],
                "leaderboard_artifact_key": leaderboard_key,
                "test_dataset_artifact_key": test_dataset_key,
                "deployment": deployment_result,
                "notebook_run": notebook_run_result,
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

        # At least 1 model produced; warn (don't fail) if fewer than top_n.
        assert len(metrics_list) >= 1, (
            f"Expected at least 1 model with metrics for scenario {func_config.id}; "
            f"found {len(metrics_list)} under s3://{artifacts_bucket}/{run_prefix}"
        )
        if len(metrics_list) < func_config.top_n:
            logger.warning(
                "Scenario %s: found %d models but top_n=%d — "
                "AutoGluon may have trained fewer models than requested on this dataset",
                func_config.id,
                len(metrics_list),
                func_config.top_n,
            )

        # Validate metrics content: all values must be numeric and the task-specific
        # primary metric must be present (guarantees evaluate_predictions ran fully).
        expected_primary_metric = _TASK_PRIMARY_METRICS.get(func_config.task_type)
        for m in metrics_list:
            model_name = m["model_name"]
            metrics = m["metrics"]
            assert len(metrics) >= 1, f"Model {model_name!r} in scenario {func_config.id} has no metrics"
            non_numeric = {k: v for k, v in metrics.items() if not isinstance(v, (int, float))}
            assert not non_numeric, f"Model {model_name!r} has non-numeric metric values: {non_numeric}"
            if expected_primary_metric:
                assert expected_primary_metric in metrics, (
                    f"Primary metric {expected_primary_metric!r} missing from model {model_name!r}; "
                    f"got keys: {sorted(metrics.keys())}"
                )

        # Validate that a notebook was written to S3 for every model.
        for m in metrics_list:
            assert m["notebook_key"] is not None, (
                f"Notebook automl_predictor_notebook.ipynb not found in S3 "
                f"for model {m['model_name']!r} in scenario {func_config.id}"
            )

        # Validate leaderboard HTML artifact from leaderboard_evaluation component.
        assert leaderboard_key is not None, (
            f"Leaderboard HTML artifact not found in S3 under s3://{artifacts_bucket}/{run_prefix}"
        )
        assert leaderboard_html and "<html" in leaderboard_html.lower(), (
            f"Leaderboard HTML at s3://{artifacts_bucket}/{leaderboard_key} is empty or has no <html> element"
        )

        # Validate sampled_test_dataset CSV artifact from automl_data_loader.
        assert test_dataset_key is not None, (
            f"sampled_test_dataset artifact not found in S3 under s3://{artifacts_bucket}/{run_prefix}"
        )

        if os.environ.get("RHOAI_NOTEBOOK_RUNNER_IMAGE", "").strip() and not notebook_run_result.get("skipped"):
            assert notebook_run_result.get("succeeded"), (
                f"Notebook execution Job failed for scenario {func_config.id}: {notebook_run_result.get('error')}"
            )

        if DEPLOY_AFTER_TRAINING and not deployment_result.get("skipped"):
            assert deployment_result.get("scored"), (
                f"Scoring failed for scenario {func_config.id}: {deployment_result.get('score_error')}"
            )
            predictions = deployment_result.get("predictions")
            assert predictions is not None, f"No predictions returned for scenario {func_config.id}"
            assert isinstance(predictions, list) and len(predictions) > 0, (
                f"Predictions for scenario {func_config.id} must be a non-empty list, got: {predictions!r}"
            )
            assert all(isinstance(p, (int, float)) for p in predictions), (
                f"Predictions for scenario {func_config.id} must be a list of ints or floats, got: {predictions!r}"
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
        run_id: str,
        rhoai_integration_config: dict,
        temp_kubeconfig_path: str | None,
    ) -> dict:
        """Deploy the top-1 model from a completed pipeline run via KServe.

        Steps:
          a. Find the top-1 model's predictor prefix in S3.
          b. Create a temporary RHOAI Data Connection secret with S3 credentials.
          c. Optionally create the ServingRuntime (if ``RHOAI_CREATE_SERVING_RUNTIME=true``).
          d. Create the InferenceService with ``storage.key`` pointing at the secret.
          e. Wait for the InferenceService to become ready.
          f. Score the model with retries if ``func_config.inference_sample`` is provided.
          g. Clean up the InferenceService in a ``finally`` block.

        Returns a dict merged into the scenario report under the ``"deployment"`` key.
        """
        try:
            from kubernetes import client
        except ImportError:
            logger.warning("kubernetes package not installed; skipping deployment test (pip install kubernetes)")
            return {"skipped": True, "reason": "kubernetes package not installed"}

        namespace = rhoai_integration_config["rhoai_project"]
        token = rhoai_integration_config.get("rhoai_token")
        serving_image = os.environ.get("RHOAI_SERVING_IMAGE", "").strip()
        create_runtime = os.environ.get("RHOAI_CREATE_SERVING_RUNTIME", "").strip().lower() in ("1", "true", "yes")
        hardware_profile_name = os.environ.get("RHOAI_HARDWARE_PROFILE_NAME", "default-profile").strip()
        hardware_profile_namespace = os.environ.get(
            "RHOAI_HARDWARE_PROFILE_NAMESPACE", "redhat-ods-applications"
        ).strip()

        # Pick top-1 model (first entry returned by _collect_model_metrics_and_sizes)
        top_model = metrics_list[0]
        model_name = top_model["model_name"]

        # ServingRuntime and InferenceService share the same name so that the
        # ISVC spec.predictor.model.runtime field is self-consistent and each
        # test run gets an isolated, uniquely named pair of resources.
        isvc_name = _make_isvc_name(func_config.id, run_id)
        # Allow overriding the ServingRuntime name (e.g. a cluster-wide runtime pre-installed
        # by the admin).  When set, the runtime is assumed to exist and is never created or
        # deleted by the test.  When unset, the test generates a per-run name (= isvc_name)
        # and creates/deletes it when RHOAI_CREATE_SERVING_RUNTIME=true.
        existing_runtime_name = os.environ.get("RHOAI_SERVING_RUNTIME_NAME", "").strip()
        serving_runtime_name = existing_runtime_name or isvc_name

        result: dict = {
            "model_name": model_name,
            "serving_runtime": serving_runtime_name,
            "storage_key": None,
            "isvc_ready": False,
            "isvc_url": None,
            "scored": False,
            "predictions": None,
            "score_error": None,
        }

        # a. Locate the predictor directory in S3 and verify the model files are present.
        predictor_prefix = _find_top_model_predictor_prefix(s3_client, artifacts_bucket, run_prefix, model_name)
        if predictor_prefix is None:
            logger.warning(
                "Could not find predictor prefix for model %r under %s/%s",
                model_name,
                artifacts_bucket,
                run_prefix,
            )
            result["score_error"] = f"Predictor prefix not found for model {model_name!r}"
            return result

        # Verify predictor.pkl exists before creating the deployment — guards against
        # the component writing files via S3-FUSE that were not yet fully flushed/uploaded
        # to S3 by the time the test reaches this point.
        predictor_objects = _list_s3_objects(s3_client, artifacts_bucket, predictor_prefix)
        predictor_keys = {obj["Key"].split("/")[-1] for obj in predictor_objects}
        if "predictor.pkl" not in predictor_keys:
            msg = (
                f"predictor.pkl not found under s3://{artifacts_bucket}/{predictor_prefix} "
                f"(found: {sorted(predictor_keys) or 'nothing'}). "
                "The pipeline component may not have finished writing model files to S3."
            )
            logger.error(msg)
            result["score_error"] = msg
            return result
        logger.info(
            "Verified %d files under s3://%s/%s before deploying",
            len(predictor_objects),
            artifacts_bucket,
            predictor_prefix,
        )

        # storage_path is the bucket-relative S3 key ending with /predictor/ (trailing
        # slash so the KServe storage initializer treats it as a directory prefix).
        # RHOAI resolves the bucket from the Data Connection secret (storage.key).
        storage_path = predictor_prefix
        result["storage_path"] = f"s3://{artifacts_bucket}/{storage_path}"
        logger.info("Deploying model %r from s3://%s/%s", model_name, artifacts_bucket, storage_path)

        result["isvc_name"] = isvc_name
        isvc_created = False
        temp_secret_name: str | None = None
        temp_sa_name: str | None = None
        temp_rbac_name: str | None = None
        temp_runtime_name: str | None = None  # set only when we create a per-run ServingRuntime

        try:
            _load_k8s_config(temp_kubeconfig_path)
            v1 = client.CoreV1Api()
            rbac_v1 = client.RbacAuthorizationV1Api()
            apps_v1 = client.AppsV1Api()
            co = client.CustomObjectsApi()

            # b. Resolve the RHOAI Data Connection (storage key) for the InferenceService.
            #
            # Preferred path: set RHOAI_KSERVE_STORAGE_KEY to the name of an existing
            # RHOAI Data Connection secret created via the Dashboard.  odh-model-controller
            # will already have this secret + companion SA in its informer cache, so it
            # processes the connection immediately when the ISVC is created.
            #
            # Fallback path (RHOAI_KSERVE_STORAGE_KEY not set): create a temporary Data
            # Connection secret + SA programmatically, wait for the controller to index
            # them, and delete them in the finally block.  NOTE: this path often fails
            # because odh-model-controller validates newly created secrets differently
            # from Dashboard-registered ones (SSA field manager conflicts, informer lag,
            # or additional secret annotations added by the Dashboard).
            existing_storage_key = os.environ.get("RHOAI_KSERVE_STORAGE_KEY", "").strip()
            if existing_storage_key:
                storage_key = existing_storage_key
                result["storage_key"] = storage_key
                logger.info(
                    "Using existing RHOAI Data Connection %r (RHOAI_KSERVE_STORAGE_KEY) — no temporary secret created",
                    storage_key,
                )
            else:
                # Create a temporary RHOAI Data Connection secret + companion SA/RBAC.
                # Both are deleted in the finally block.
                temp_secret_name = f"kserve-s3-{isvc_name[:40]}"
                _create_kserve_s3_secret(v1, namespace, temp_secret_name, artifacts_bucket, rhoai_integration_config)
                storage_key = temp_secret_name
                result["storage_key"] = storage_key

                # Confirm the secret was stored with the correct keys (log keys, not values).
                created_secret = v1.read_namespaced_secret(
                    temp_secret_name, namespace, _request_timeout=_K8S_CALL_TIMEOUT
                )
                secret_keys = sorted((created_secret.data or {}).keys())
                logger.info(
                    "Created RHOAI Data Connection secret %r (bucket=%r, keys=%s)",
                    storage_key,
                    artifacts_bucket,
                    secret_keys,
                )

                temp_sa_name = _create_connection_sa(v1, namespace, temp_secret_name)
                # Grant the SA GET access to the secret so the KServe agent sidecar can
                # read S3 credentials from the Kubernetes API (the agent authenticates
                # with the pod's SA token).  Without this the agent gets HTTP 403,
                # downloads nothing, and the kserve-container crashes with
                # 'predictor.pkl not found'.  RHOAI Dashboard creates equivalent RBAC
                # when registering a Data Connection.
                temp_rbac_name = _create_connection_rbac(rbac_v1, namespace, temp_sa_name, temp_secret_name)

                # Wait for odh-model-controller's informer cache to index the new secret
                # and SA before creating the ISVC.  The controller reconciles the ISVC once
                # on creation; if the secret/SA are not yet in the cache at that moment it
                # skips storage setup and does NOT re-reconcile when they appear (it only
                # watches ISVC events, not Secret/SA events).  A brief pause here is enough
                # for the watch notification to arrive and the informer to update its cache.
                informer_settle_seconds = 15
                logger.info(
                    "Waiting %ds for controller informer to index secret %r and SA %r...",
                    informer_settle_seconds,
                    temp_secret_name,
                    temp_sa_name,
                )
                time.sleep(informer_settle_seconds)

            # c. ServingRuntime (optional creation)
            # RHOAI_SERVING_RUNTIME_NAME controls the runtime *name* only.
            # Creation is still governed by RHOAI_CREATE_SERVING_RUNTIME=true.
            if create_runtime:
                if not serving_image:
                    logger.warning(
                        "RHOAI_CREATE_SERVING_RUNTIME=true but RHOAI_SERVING_IMAGE is not set; "
                        "skipping runtime creation — ServingRuntime %r must already exist",
                        serving_runtime_name,
                    )
                else:
                    runtime_newly_created = _ensure_serving_runtime(
                        co,
                        namespace,
                        serving_runtime_name,
                        serving_image,
                    )
                    if runtime_newly_created:
                        # Track per-run runtimes for cleanup; shared runtimes
                        # (RHOAI_SERVING_RUNTIME_NAME set) are never deleted.
                        if not existing_runtime_name:
                            temp_runtime_name = serving_runtime_name
                        # KServe's controller needs time to index a freshly created
                        # ServingRuntime. If we submit the ISVC immediately, the
                        # reconciler may not find the runtime yet and back off
                        # exponentially before ever creating the predictor Deployment.
                        settle_seconds = 30
                        logger.info(
                            "ServingRuntime %r was just created — waiting %ds for KServe "
                            "controller to index it before creating the InferenceService...",
                            serving_runtime_name,
                            settle_seconds,
                        )
                        time.sleep(settle_seconds)

            # d. Create InferenceService
            # Fetch hardware profile resource version — odh-model-controller requires
            # this annotation to match the live HardwareProfile resourceVersion before
            # it will create the predictor Deployment (Route/Service are created by
            # KServe regardless). Allow override via env var for offline/air-gapped use.
            hw_rv = os.environ.get("RHOAI_HARDWARE_PROFILE_RESOURCE_VERSION", "").strip()
            if not hw_rv:
                hw_rv = _fetch_hardware_profile_resource_version(co, hardware_profile_namespace, hardware_profile_name)
            if not hw_rv:
                raise RuntimeError(
                    f"Could not resolve opendatahub.io/hardware-profile-resource-version for "
                    f"HardwareProfile {hardware_profile_name!r} in namespace {hardware_profile_namespace!r}. "
                    "odh-model-controller will not create the predictor Deployment without this annotation "
                    "on the InferenceService. Grant get/list on hardwareprofiles.infrastructure.opendatahub.io "
                    "if needed, fix RHOAI_HARDWARE_PROFILE_NAME / RHOAI_HARDWARE_PROFILE_NAMESPACE, or set "
                    "RHOAI_HARDWARE_PROFILE_RESOURCE_VERSION to "
                    "`oc get hardwareprofile -n <ns> <name> -o jsonpath='{.metadata.resourceVersion}'`."
                )
            # d. Create the ISVC. The controller has already indexed the secret
            # (SA confirmed above), so it will set storage.key / serviceAccountName
            # in the same reconcile loop that creates the Deployment.
            _create_inference_service(
                co,
                namespace,
                isvc_name,
                serving_runtime_name,
                storage_path,
                storage_key,
                hardware_profile_name=hardware_profile_name,
                hardware_profile_namespace=hardware_profile_namespace,
                hardware_profile_resource_version=hw_rv,
            )
            isvc_created = True
            logger.info("Created InferenceService %r in namespace %r", isvc_name, namespace)

            # e. Ensure the predictor Deployment has the storage initializer annotations.
            # odh-model-controller normally sets these when it processes the
            # opendatahub.io/connections annotation on the ISVC.  If connection
            # processing is silently skipped (informer lag, secret format mismatch,
            # etc.) the function patches the Deployment directly so the pods can
            # still download the model on their next restart.
            logger.info(
                "Waiting for predictor Deployment %r-predictor and checking storage annotations "
                "(polls every 5s, timeout 60s)...",
                isvc_name,
            )
            _ensure_deployment_storage_annotations(
                apps_v1,
                namespace,
                isvc_name,
                storage_key=storage_key,
                artifacts_bucket=artifacts_bucket,
                storage_path=storage_path,
                wait_seconds=60,
            )
            _log_isvc_events(v1, namespace, isvc_name)

            # f. Wait for InferenceService to become Ready (polls every 30 s, logging
            # all conditions and controller-populated spec fields so blocking errors
            # surface immediately rather than after a silent timeout).
            # RHOAI_INFERENCE_TIMEOUT controls the total wait (default: 300 s).
            inference_timeout = int(os.environ.get("RHOAI_INFERENCE_TIMEOUT", "300"))
            logger.info(
                "InferenceService %r created — waiting up to %ds for Ready=True "
                "(polling every 30s with full condition logging)...",
                isvc_name,
                inference_timeout,
            )
            isvc_ready, blocking_reason = _wait_for_isvc_ready(
                co, namespace, isvc_name, timeout_seconds=inference_timeout
            )
            result["isvc_ready"] = isvc_ready

            if blocking_reason:
                msg = f"InferenceService {isvc_name!r} has a blocking condition: {blocking_reason}"
                logger.error(msg)
                result["score_error"] = msg
                return result

            if not isvc_ready:
                logger.warning(
                    "InferenceService %r not Ready after %ds — "
                    "Route/Service may still exist; attempting score with retries...",
                    isvc_name,
                    inference_timeout,
                )

            external_url = _resolve_isvc_external_url(co, namespace, isvc_name)
            result["isvc_url"] = external_url

            if not external_url:
                msg = (
                    f"InferenceService {isvc_name!r}: no external Route host found "
                    f"after {inference_timeout}s — cannot score from outside the cluster. "
                    f"Ensure the ISVC has networking.kserve.io/visibility=exposed."
                )
                logger.warning(msg)
                result["score_error"] = msg
                return result

            logger.info("InferenceService %r: scoring via %s", isvc_name, external_url)

            # g. Score via the external Route URL with retries.
            if not func_config.inference_sample:
                logger.info("No inference_sample in config %r — skipping scoring", func_config.id)
            else:
                instances = _column_sample_to_instances(func_config.inference_sample)
                logger.info(
                    "Scoring %r via %s with %d instance(s): %s",
                    isvc_name,
                    external_url,
                    len(instances),
                    json.dumps(instances, default=str),
                )
                try:
                    response = _score_inference_service(external_url, isvc_name, instances, token)
                    result["scored"] = True
                    result["predictions"] = response.get("predictions")
                    logger.info("Scoring succeeded for %r: %s", isvc_name, json.dumps(response, default=str))
                except Exception as score_err:
                    pod_logs = _fetch_pod_logs_str(v1, namespace, f"serving.kserve.io/inferenceservice={isvc_name}")
                    logger.warning("Scoring failed for %r: %s", isvc_name, score_err)
                    result["score_error"] = f"{score_err}\n{pod_logs}"

        except Exception as deploy_err:
            logger.error("Deployment test failed for scenario %r: %s", func_config.id, deploy_err, exc_info=True)
            result["score_error"] = str(deploy_err)

        finally:
            # h. Always clean up all created Kubernetes objects — unconditionally.
            # AUTOML_FUNCTIONAL_TEST_KEEP_ARTIFACTS only governs S3 artifact retention;
            # it does NOT affect teardown of deployment resources (ISVC, secret, SA, RBAC).
            # These are always deleted here to avoid orphaning resources on the cluster.
            if isvc_created:
                try:
                    _load_k8s_config(temp_kubeconfig_path)
                    co_cleanup = client.CustomObjectsApi()
                    _delete_inference_service(co_cleanup, namespace, isvc_name)
                    logger.info("Deleted InferenceService %r", isvc_name)
                except Exception as cleanup_err:
                    logger.warning("Failed to clean up InferenceService %r: %s", isvc_name, cleanup_err)
            # Clean up a per-run ServingRuntime if we created one.
            # Shared runtimes (RHOAI_SERVING_RUNTIME_NAME set) are never deleted.
            if temp_runtime_name:
                try:
                    co.delete_namespaced_custom_object(
                        group=_KSERVE_GROUP,
                        version=_KSERVE_SR_VERSION,
                        namespace=namespace,
                        plural=_KSERVE_SR_PLURAL,
                        name=temp_runtime_name,
                        _request_timeout=_K8S_CALL_TIMEOUT,
                    )
                    logger.info("Deleted temporary ServingRuntime %r", temp_runtime_name)
                except Exception as sr_err:
                    logger.warning("Failed to delete temporary ServingRuntime %r: %s", temp_runtime_name, sr_err)
            # Clean up the temporary S3 secret if we created one
            if temp_secret_name:
                try:
                    v1.delete_namespaced_secret(temp_secret_name, namespace, _request_timeout=_K8S_CALL_TIMEOUT)
                    logger.info("Deleted temporary KServe S3 secret %r", temp_secret_name)
                except Exception as secret_cleanup_err:
                    logger.warning("Failed to delete temporary secret %r: %s", temp_secret_name, secret_cleanup_err)
            # Clean up the temporary RBAC (Role + RoleBinding) if we created them
            if temp_rbac_name:
                try:
                    rbac_v1.delete_namespaced_role_binding(
                        temp_rbac_name, namespace, _request_timeout=_K8S_CALL_TIMEOUT
                    )
                    logger.info("Deleted temporary RoleBinding %r", temp_rbac_name)
                except Exception as rb_err:
                    logger.warning("Failed to delete temporary RoleBinding %r: %s", temp_rbac_name, rb_err)
                try:
                    rbac_v1.delete_namespaced_role(temp_rbac_name, namespace, _request_timeout=_K8S_CALL_TIMEOUT)
                    logger.info("Deleted temporary Role %r", temp_rbac_name)
                except Exception as role_err:
                    logger.warning("Failed to delete temporary Role %r: %s", temp_rbac_name, role_err)
            # Clean up the temporary companion ServiceAccount if we created one
            if temp_sa_name:
                try:
                    v1.delete_namespaced_service_account(temp_sa_name, namespace, _request_timeout=_K8S_CALL_TIMEOUT)
                    logger.info("Deleted temporary ServiceAccount %r", temp_sa_name)
                except Exception as sa_cleanup_err:
                    logger.warning("Failed to delete temporary SA %r: %s", temp_sa_name, sa_cleanup_err)

        return result

    # ------------------------------------------------------------------
    # Notebook execution helper (called from test_scenario when enabled)
    # ------------------------------------------------------------------

    def _run_notebook_test(
        self,
        *,
        notebook_key: str | None,
        run_id: str,
        model_name: str,
        s3_client,
        artifacts_bucket: str,
        rhoai_integration_config: dict,
        temp_kubeconfig_path: str | None,
    ) -> dict:
        """Execute the top-1 model notebook as a Kubernetes Job using papermill.

        Requires ``RHOAI_NOTEBOOK_RUNNER_IMAGE`` to be set to a container image that
        has both ``papermill`` and ``autogluon`` installed. When unset the step is skipped.

        Steps:
          a. Download the notebook JSON from S3 (``notebook_key``).
          b. Create a Kubernetes ConfigMap with the notebook content.
          c. Create a Kubernetes Job running ``papermill`` against the notebook.
          d. Poll until the Job succeeds, fails, or times out.
          e. Delete Job + ConfigMap in a ``finally`` block.

        Returns a dict with:
            ``succeeded``: bool — True only when the Job's ``status.succeeded`` is 1.
            ``skipped``:   bool — True when the step was skipped (no image / no notebook).
            ``reason``:    str | None — why the step was skipped.
            ``error``:     str | None — error message when not succeeded.
            ``job_name``:  str | None — Kubernetes Job name (for debugging).
            ``elapsed_seconds``: float | None — wall time from Job creation to completion.
        """
        runner_image = os.environ.get("RHOAI_NOTEBOOK_RUNNER_IMAGE", "").strip()
        if not runner_image:
            logger.info("RHOAI_NOTEBOOK_RUNNER_IMAGE not set — skipping notebook execution test")
            return {"skipped": True, "reason": "RHOAI_NOTEBOOK_RUNNER_IMAGE not set"}

        if not notebook_key:
            logger.warning("notebook_key is None for model %r — skipping notebook execution test", model_name)
            return {"skipped": True, "reason": f"notebook not found in S3 for model {model_name!r}"}

        try:
            from kubernetes import client as k8s_client
            from kubernetes.client.rest import ApiException as K8sApiException
        except ImportError:
            logger.warning("kubernetes package not installed; skipping notebook execution test")
            return {"skipped": True, "reason": "kubernetes package not installed"}

        notebook_run_timeout = int(os.environ.get("RHOAI_NOTEBOOK_RUN_TIMEOUT", str(_NOTEBOOK_JOB_TIMEOUT)))
        namespace = rhoai_integration_config["rhoai_project"]
        job_name = _make_notebook_job_name(run_id, model_name)

        result: dict = {
            "succeeded": False,
            "skipped": False,
            "reason": None,
            "error": None,
            "job_name": job_name,
            "elapsed_seconds": None,
        }

        # a. Download the notebook JSON from S3.
        try:
            resp = s3_client.get_object(Bucket=artifacts_bucket, Key=notebook_key)
            notebook_content = resp["Body"].read().decode("utf-8")
        except Exception as exc:
            msg = f"Failed to download notebook from s3://{artifacts_bucket}/{notebook_key}: {exc}"
            logger.error(msg)
            result["error"] = msg
            return result

        # ConfigMap has a 1 MB practical limit; skip gracefully if the notebook is too large.
        notebook_size_bytes = len(notebook_content.encode("utf-8"))
        if notebook_size_bytes > 900_000:
            msg = (
                f"Notebook s3://{artifacts_bucket}/{notebook_key} is {notebook_size_bytes // 1024} KB "
                "(> 900 KB) — too large for a Kubernetes ConfigMap; skipping notebook execution test"
            )
            logger.warning(msg)
            return {"skipped": True, "reason": msg}

        logger.info(
            "Downloaded notebook from s3://%s/%s (%d KB) for model %r",
            artifacts_bucket,
            notebook_key,
            notebook_size_bytes // 1024,
            model_name,
        )

        job_created = False
        cm_created = False
        try:
            _load_k8s_config(temp_kubeconfig_path)
            v1 = k8s_client.CoreV1Api()
            batch_v1 = k8s_client.BatchV1Api()

            # b. Create ConfigMap with notebook content (key: notebook.ipynb → /input/notebook.ipynb).
            configmap = k8s_client.V1ConfigMap(
                metadata=k8s_client.V1ObjectMeta(name=job_name, namespace=namespace),
                data={"notebook.ipynb": notebook_content},
            )
            try:
                v1.create_namespaced_config_map(namespace, configmap, _request_timeout=_K8S_CALL_TIMEOUT)
                cm_created = True
                logger.info("Created ConfigMap %r for notebook execution", job_name)
            except K8sApiException as exc:
                if exc.status == 409:
                    v1.replace_namespaced_config_map(job_name, namespace, configmap, _request_timeout=_K8S_CALL_TIMEOUT)
                    cm_created = True
                    logger.info("Replaced existing ConfigMap %r", job_name)
                else:
                    raise

            # c. Create a Kubernetes Job running papermill against the notebook.
            env_vars = [
                k8s_client.V1EnvVar(name="AWS_ACCESS_KEY_ID", value=rhoai_integration_config["s3_access_key"]),
                k8s_client.V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value=rhoai_integration_config["s3_secret_key"]),
                k8s_client.V1EnvVar(name="AWS_S3_ENDPOINT", value=rhoai_integration_config["s3_endpoint"]),
                k8s_client.V1EnvVar(
                    name="AWS_DEFAULT_REGION",
                    value=rhoai_integration_config.get("s3_region", "us-east-1"),
                ),
                k8s_client.V1EnvVar(name="AWS_S3_BUCKET", value=artifacts_bucket),
                # Redirect all pip installs (setup and notebook cells) to /tmp so they
                # succeed regardless of whether the venv is writable by the runtime UID.
                # HOME=/tmp lets ipykernel --user and Jupyter find the registered kernel.
                # PIP_INDEX_URL overrides pip.conf so the image's internal Red Hat mirror
                # (console.redhat.com) is bypassed in favour of the public PyPI.
                # k8s_client.V1EnvVar(name="HOME", value="/tmp"),
                # k8s_client.V1EnvVar(name="PIP_TARGET", value="/tmp/nb-packages"),
                # k8s_client.V1EnvVar(name="PYTHONPATH", value="/tmp/nb-packages"),
                # k8s_client.V1EnvVar(name="PIP_INDEX_URL", value="https://pypi.org/simple/"),
                k8s_client.V1EnvVar(name="PIP_RETRIES", value="2"),
            ]
            job = k8s_client.V1Job(
                metadata=k8s_client.V1ObjectMeta(name=job_name, namespace=namespace),
                spec=k8s_client.V1JobSpec(
                    backoff_limit=0,
                    ttl_seconds_after_finished=3600,
                    template=k8s_client.V1PodTemplateSpec(
                        spec=k8s_client.V1PodSpec(
                            restart_policy="Never",
                            containers=[
                                k8s_client.V1Container(
                                    name="notebook-runner",
                                    image=runner_image,
                                    command=["/bin/sh", "-c"],
                                    args=[
                                        "mkdir -p /tmp/nb-packages "
                                        "&& pip install --quiet papermill ipykernel "
                                        "&& python -m ipykernel install --user --name python3 "
                                        "&& python -m papermill /input/notebook.ipynb /dev/null "
                                        "--no-progress-bar --log-output"
                                    ],
                                    env=env_vars,
                                    volume_mounts=[
                                        k8s_client.V1VolumeMount(
                                            name="notebook",
                                            mount_path="/input",
                                        )
                                    ],
                                    security_context=k8s_client.V1SecurityContext(
                                        allow_privilege_escalation=False,
                                        run_as_non_root=True,
                                        capabilities=k8s_client.V1Capabilities(drop=["ALL"]),
                                    ),
                                )
                            ],
                            volumes=[
                                k8s_client.V1Volume(
                                    name="notebook",
                                    config_map=k8s_client.V1ConfigMapVolumeSource(name=job_name),
                                )
                            ],
                        )
                    ),
                ),
            )
            try:
                batch_v1.create_namespaced_job(namespace, job, _request_timeout=_K8S_CALL_TIMEOUT)
                job_created = True
                logger.info("Created notebook execution Job %r (image=%r)", job_name, runner_image)
            except K8sApiException as exc:
                if exc.status != 409:
                    raise

            # d. Poll until Job completes or times out.
            start = time.monotonic()
            poll_interval = 15  # seconds
            while True:
                elapsed = time.monotonic() - start
                try:
                    j = batch_v1.read_namespaced_job(job_name, namespace, _request_timeout=_K8S_CALL_TIMEOUT)
                    succeeded = j.status.succeeded or 0
                    failed = j.status.failed or 0
                    active = j.status.active or 0
                    logger.info(
                        "Job %r: succeeded=%d failed=%d active=%d (elapsed %.0fs)",
                        job_name,
                        succeeded,
                        failed,
                        active,
                        elapsed,
                    )
                    if succeeded >= 1:
                        result["succeeded"] = True
                        result["elapsed_seconds"] = round(elapsed, 1)
                        logger.info("Notebook Job %r succeeded in %.0fs", job_name, elapsed)
                        break
                    if failed >= 1:
                        pod_logs = _fetch_pod_logs_str(v1, namespace, f"job-name={job_name}")
                        msg = f"Notebook Job {job_name!r} failed after {elapsed:.0f}s\n{pod_logs}"
                        logger.error(msg)
                        result["error"] = msg
                        result["elapsed_seconds"] = round(elapsed, 1)
                        break
                except Exception as poll_exc:
                    logger.warning("Failed to poll Job %r: %s", job_name, poll_exc)

                if elapsed >= notebook_run_timeout:
                    pod_logs = _fetch_pod_logs_str(v1, namespace, f"job-name={job_name}")
                    msg = f"Notebook Job {job_name!r} timed out after {elapsed:.0f}s\n{pod_logs}"
                    logger.error(msg)
                    result["error"] = msg
                    result["elapsed_seconds"] = round(elapsed, 1)
                    break

                sleep_secs = min(poll_interval, max(1, notebook_run_timeout - elapsed))
                logger.info(
                    "Job %r: next poll in %.0fs (%.0fs remaining)...",
                    job_name,
                    sleep_secs,
                    max(0, notebook_run_timeout - elapsed),
                )
                time.sleep(sleep_secs)

        except Exception as exc:
            msg = f"Notebook execution test failed for model {model_name!r}: {exc}"
            logger.error(msg, exc_info=True)
            result["error"] = msg

        finally:
            # e. Clean up Job and ConfigMap unconditionally.
            if job_created:
                try:
                    _load_k8s_config(temp_kubeconfig_path)
                    batch_v1_cleanup = k8s_client.BatchV1Api()
                    batch_v1_cleanup.delete_namespaced_job(
                        job_name,
                        namespace,
                        body=k8s_client.V1DeleteOptions(propagation_policy="Background"),
                        _request_timeout=_K8S_CALL_TIMEOUT,
                    )
                    logger.info("Deleted notebook Job %r", job_name)
                except Exception as cleanup_err:
                    logger.warning("Failed to delete notebook Job %r: %s", job_name, cleanup_err)
            if cm_created:
                try:
                    _load_k8s_config(temp_kubeconfig_path)
                    v1_cleanup = k8s_client.CoreV1Api()
                    v1_cleanup.delete_namespaced_config_map(job_name, namespace, _request_timeout=_K8S_CALL_TIMEOUT)
                    logger.info("Deleted notebook ConfigMap %r", job_name)
                except Exception as cleanup_err:
                    logger.warning("Failed to delete notebook ConfigMap %r: %s", job_name, cleanup_err)
        return result
