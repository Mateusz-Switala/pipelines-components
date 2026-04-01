"""Functional tests for AutoGluon time series training pipeline (TC-B matrix).

These tests exercise the full end-to-end pipeline on a real RHOAI cluster with
real datasets. Each scenario is driven by a JSON config file (default:
functional_test_configs.json). Scenarios are parametrized so pytest runs each
independently.

Environment variables (on top of the standard integration .env):
    AUTOML_FUNCTIONAL_TEST_CONFIG  — path to the JSON config file
                              (default: functional_test_configs.json in this dir)
    AUTOML_FUNCTIONAL_TEST_REPORT  — path to the JSON report output
                              (default: functional_test_report.json in this dir)

Parallel execution:
    Use pytest-xdist to run scenarios concurrently by passing ``-n <workers>``
    to pytest (e.g. ``-n 3``).

The test flow for each scenario:
    1. Read scenario configuration from the config file
    2. Preprocess the dataset (add dummy item_id / timestamp if needed)
    3. Upload the dataset to S3
    4. Run the pipeline with the correct input arguments
    5. Wait until the pipeline run completes successfully
    6. Measure wall-clock time for the run
    7. Read metrics for top_n models from the S3 artifacts
    8. Measure the total size of resulting model artifacts in S3
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

import csv
import json
import logging
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import StringIO
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

from ..pipeline import autogluon_timeseries_training_pipeline  # noqa: E402

PIPELINE_DISPLAY_NAME = autogluon_timeseries_training_pipeline.name


# ---------------------------------------------------------------------------
# Functional test config dataclass
# ---------------------------------------------------------------------------


@dataclass
class FunctionalTestConfig:
    """Single functional test scenario for the time series pipeline."""

    id: str
    dataset_path: str
    target: str
    id_column: str
    timestamp_column: str
    known_covariates_names: list[str]
    prediction_length: int
    top_n: int
    train_data_file_key: str
    add_dummy_item_id: bool = False
    add_dummy_timestamp: bool = False
    tags: list[str] = field(default_factory=list)

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
            "target": self.target,
            "id_column": self.id_column,
            "timestamp_column": self.timestamp_column,
            "known_covariates_names": self.known_covariates_names,
            "prediction_length": self.prediction_length,
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
                target=item["target"],
                id_column=item["id_column"],
                timestamp_column=item["timestamp_column"],
                known_covariates_names=item.get("known_covariates_names", []),
                prediction_length=int(item.get("prediction_length", 1)),
                top_n=int(item.get("top_n", 3)),
                train_data_file_key=item["train_data_file_key"],
                add_dummy_item_id=item.get("add_dummy_item_id", False),
                add_dummy_timestamp=item.get("add_dummy_timestamp", False),
                tags=item.get("tags", []),
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


# ---------------------------------------------------------------------------
# Dataset preprocessing
# ---------------------------------------------------------------------------


def _preprocess_csv(
    local_path: Path,
    add_dummy_item_id: bool,
    add_dummy_timestamp: bool,
    item_id_column: str = "item_id",
    timestamp_column: str = "timestamp",
) -> bytes:
    """Read a CSV and optionally add dummy item_id / timestamp columns.

    Returns the (possibly modified) CSV content as bytes ready for S3 upload.

    - add_dummy_item_id: inserts a constant ``item_id_column`` column with value "ts_0".
    - add_dummy_timestamp: generates a sequential ``timestamp_column`` column
      starting at 2020-01-01 with daily frequency.
    """
    text = local_path.read_text(encoding="utf-8")
    if not add_dummy_item_id and not add_dummy_timestamp:
        return text.encode("utf-8")

    reader = csv.reader(StringIO(text))
    header = next(reader)
    rows = list(reader)

    if add_dummy_item_id:
        header.insert(0, item_id_column)
        for row in rows:
            row.insert(0, "ts_0")

    if add_dummy_timestamp:
        header.insert(0, timestamp_column)
        from datetime import timedelta

        base = datetime(2020, 1, 1)
        for i, row in enumerate(rows):
            row.insert(0, (base + timedelta(days=i)).strftime("%Y-%m-%d"))

    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_name() -> str:
    hex_part = secrets.token_hex(3)
    time_part = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"ts-functional-{hex_part}-{time_part}"


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
    return [obj for page in paginator.paginate(Bucket=bucket, Prefix=prefix) for obj in page.get("Contents") or []]


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
    keep_artifacts = os.environ.get("AUTOML_FUNCTIONAL_TEST_KEEP_ARTIFACTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
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
class TestTimeseriesPipelineFunctional:
    """Functional tests running TC-B scenarios on a real RHOAI cluster."""

    def test_scenario(
        self,
        func_config: FunctionalTestConfig,
        rhoai_integration_config,
        rhoai_project,  # noqa: ARG002 — ensures project/secret exist before runs
        kfp_client,
        compiled_pipeline_path,
        s3_client,
        s3_cleanup_tracker: S3CleanupTracker,
    ):
        """Run one TC-B scenario end-to-end and validate results."""
        if not kfp_client:
            pytest.skip("KFP client not available")

        config = rhoai_integration_config
        data_bucket = config["s3_bucket_data"]
        artifacts_bucket = config.get("s3_bucket_artifacts") or data_bucket
        secret_name = config["s3_secret_name"]

        # ------------------------------------------------------------------
        # 1. Preprocess and upload dataset to S3
        # ------------------------------------------------------------------
        dataset_local_path = _TESTS_DIR / func_config.dataset_path
        if not dataset_local_path.is_file():
            pytest.fail(f"Dataset file not found: {dataset_local_path}")

        # Preprocess: add dummy columns where needed
        csv_bytes = _preprocess_csv(
            dataset_local_path,
            add_dummy_item_id=func_config.add_dummy_item_id,
            add_dummy_timestamp=func_config.add_dummy_timestamp,
            item_id_column=func_config.id_column,
            timestamp_column=func_config.timestamp_column,
        )

        # Count rows and features from the preprocessed CSV
        csv_text = csv_bytes.decode("utf-8")
        reader = csv.reader(StringIO(csv_text))
        header = next(reader)
        num_columns = len(header)
        num_rows = sum(1 for _ in reader)
        num_features = num_columns - 1  # exclude target column

        logger.info(
            "Dataset %s: %d rows, %d features (+ target '%s')",
            dataset_local_path.name,
            num_rows,
            num_features,
            func_config.target,
        )
        if func_config.add_dummy_item_id:
            logger.info("  Added dummy item_id column '%s'", func_config.id_column)
        if func_config.add_dummy_timestamp:
            logger.info("  Added dummy timestamp column '%s'", func_config.timestamp_column)

        s3_key = func_config.train_data_file_key
        logger.info("Uploading %s -> s3://%s/%s", dataset_local_path.name, data_bucket, s3_key)
        s3_client.put_object(
            Bucket=data_bucket,
            Key=s3_key,
            Body=csv_bytes,
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
            "Submitting pipeline run %s for scenario %s (target=%s, top_n=%d, prediction_length=%d)",
            run_name,
            func_config.id,
            func_config.target,
            func_config.top_n,
            func_config.prediction_length,
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
        # 6. Write per-scenario report (xdist-safe: one file per scenario)
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
                "target": func_config.target,
                "id_column": func_config.id_column,
                "timestamp_column": func_config.timestamp_column,
                "known_covariates_names": func_config.known_covariates_names,
                "prediction_length": func_config.prediction_length,
                "top_n": func_config.top_n,
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
                "config": asdict(func_config),
            },
        )

        # ------------------------------------------------------------------
        # 7. Assertions
        # ------------------------------------------------------------------
        assert elapsed_seconds < MAX_RUN_DURATION_SECONDS, (
            f"Scenario {func_config.id} took {elapsed_minutes:.1f} min "
            f"({elapsed_seconds:.0f} s), exceeding the 1-hour limit"
        )

        assert len(metrics_list) >= 1, (
            f"Expected at least 1 model with metrics for scenario {func_config.id}; "
            f"found {len(metrics_list)} under s3://{artifacts_bucket}/{run_prefix}"
        )
