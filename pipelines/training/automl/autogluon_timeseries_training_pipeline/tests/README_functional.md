# Functional Tests — AutoGluon Time Series Training Pipeline (TC-B)

Functional tests for the `autogluon_timeseries_training_pipeline`. Each scenario runs the
full pipeline end-to-end on a real RHOAI cluster with real datasets.

## Scenarios

| ID | Dataset | Target | id_column | timestamp | Covariates | prediction_length | top_n |
|----|---------|--------|-----------|-----------|------------|-------------------|-------|
| TC-B-1 | retail_sales_dataset (1,000 rows) | Total Amount | Product Category | Date | — | 1 | 3 |
| TC-B-2 | traffic_dataset (10,000 rows) | vehicle_count | item_id (dummy) | timestamp (dummy) | average_speed, lane_occupancy | 14 | 1 |
| TC-B-3 | Electricity Load Forecasting (48,049 rows) | nat_demand | item_id (dummy) | datetime | — | 7 | 5 |

Scenarios are defined in `functional_test_configs.json`. You can point to a
custom config file via the `AUTOML_FUNCTIONAL_TEST_CONFIG` environment variable.

### Dummy columns

Some datasets do not have an `item_id` column (single time series) or a proper
timestamp column. The test preprocesses these CSVs before uploading to S3:

- **`add_dummy_item_id`**: inserts a constant column (e.g. `item_id = "ts_0"`)
  so AutoGluon treats the data as a single time series.
- **`add_dummy_timestamp`**: generates sequential daily dates starting at
  `2020-01-01` in a new column.

## Prerequisites

1. **RHOAI cluster** with Data Science Pipelines enabled and a
   DataSciencePipelinesApplication (DSPA) deployed.
2. **S3-compatible storage** (MinIO, AWS S3, etc.) accessible from the cluster
   and from the machine running the tests.
3. **Python environment** with test dependencies installed:

   ```bash
   pip install pytest boto3 kfp python-dotenv pyyaml kubernetes
   # or, from the repo root:
   uv sync --extra test_automl
   ```

4. **Environment variables** — copy `.env.template` to `.env` in this directory
   and fill in the values.

   **Core variables:**

   | Variable | Required | Description |
   |----------|----------|-------------|
   | `RHOAI_URL` | yes | OpenShift API server URL |
   | `RHOAI_TOKEN` | yes | Bearer token (SA or user) |
   | `RHOAI_PROJECT_NAME` | no | Namespace (default: `kfp-integration-test`) |
   | `RHOAI_KFP_URL` | conditional | KFP API URL (not needed if `RHOAI_CREATE_DSPA=true`) |
   | `AWS_S3_ENDPOINT` | yes | S3 endpoint URL |
   | `AWS_ACCESS_KEY_ID` | yes | S3 access key |
   | `AWS_SECRET_ACCESS_KEY` | yes | S3 secret key |
   | `AWS_DEFAULT_REGION` | no | S3 region (default: `us-east-1`) |
   | `RHOAI_TEST_DATA_BUCKET` | yes | Bucket for uploading test datasets |
   | `RHOAI_TEST_ARTIFACTS_BUCKET` | no | Bucket for pipeline artifacts (defaults to data bucket) |
   | `RHOAI_TEST_S3_SECRET_NAME` | no | K8s secret name (default: `s3-connection`) |
   | `RHOAI_PIPELINE_RUN_TIMEOUT` | no | Max wait time in seconds (default: `3600`) |
   | `RHOAI_CREATE_DSPA` | no | Set to `true` to auto-create DSPA |

   **Functional-test-specific variables:**

   | Variable | Required | Description |
   |----------|----------|-------------|
   | `AUTOML_FUNCTIONAL_TEST_CONFIG` | no | Path to the JSON config file (default: `functional_test_configs.json`) |
   | `AUTOML_FUNCTIONAL_TEST_REPORT` | no | Path to the JSON report output (default: `functional_test_report.json`) |
   | `AUTOML_FUNCTIONAL_TEST_KEEP_ARTIFACTS` | no | Set to `true` to skip deleting pipeline run artifacts from S3 after tests. Uploaded datasets are still cleaned up. |

   **Post-pipeline notebook execution (optional):**

   When `RHOAI_NOTEBOOK_RUNNER_IMAGE` is set, the test runs the top-1 model's predictor
   notebook as a Kubernetes Job after each pipeline run and asserts it completes
   successfully. The notebook installs `autogluon.timeseries` from the Red Hat PyPI
   mirror at runtime.

   | Variable | Required | Description |
   |----------|----------|-------------|
   | `RHOAI_NOTEBOOK_RUNNER_IMAGE` | no | Container image with `papermill` and `autogluon.timeseries` for notebook execution. When unset the step is skipped. |
   | `RHOAI_NOTEBOOK_RUN_TIMEOUT` | no | Seconds to wait for the notebook Job to finish (default: `1200`) |

   **Post-pipeline KServe deployment (optional):**

   When `RHOAI_DEPLOY_AFTER_TRAINING=true`, the test deploys the top-1 model as a
   KServe InferenceService and scores it using `inference_sample` from the config.

   | Variable | Required | Description |
   |----------|----------|-------------|
   | `RHOAI_DEPLOY_AFTER_TRAINING` | no | Set to `true` to deploy and score the top-1 model (default: `false`) |
   | `RHOAI_SERVING_IMAGE` | conditional | Container image for the AutoGluon ServingRuntime. Required when `RHOAI_CREATE_SERVING_RUNTIME=true`. |
   | `RHOAI_CREATE_SERVING_RUNTIME` | no | Set to `true` to auto-create the ServingRuntime (requires `RHOAI_SERVING_IMAGE`) |
   | `RHOAI_SERVING_RUNTIME_NAME` | no | Name of an existing ServingRuntime to use. When set, `RHOAI_CREATE_SERVING_RUNTIME` is ignored. |
   | `RHOAI_KSERVE_STORAGE_KEY` | no | Name of an existing RHOAI Data Connection secret pointing to the artifacts bucket. Recommended over letting the test create a temporary secret. |
   | `RHOAI_HARDWARE_PROFILE_NAME` | no | Hardware profile for the InferenceService (default: `default-profile`) |
   | `RHOAI_HARDWARE_PROFILE_NAMESPACE` | no | Namespace where the hardware profile lives (default: `redhat-ods-applications`) |
   | `RHOAI_HARDWARE_PROFILE_RESOURCE_VERSION` | no | Resource version of the hardware profile. Fetched automatically when unset. |
   | `RHOAI_INFERENCE_TIMEOUT` | no | Seconds to wait for InferenceService to become ready (default: `300`) |

5. **Test datasets** must be present under `tests/data/`:
   - `retail_sales_dataset.csv` — Retail sales time series
   - `traffic_dataset.csv` — Traffic flow time series
   - `Electricity Load Forecasting.csv` — Electricity demand time series

## Running the Tests

```bash
# Run all functional scenarios (sequential)
uv run pytest pipelines/training/automl/autogluon_timeseries_training_pipeline/tests/test_pipeline_functional.py \
    -m functional -v -s --log-cli-level=INFO

# Run all scenarios in parallel (3 workers via pytest-xdist)
uv run pytest pipelines/training/automl/autogluon_timeseries_training_pipeline/tests/test_pipeline_functional.py \
    -m functional -v -s --log-cli-level=INFO -n 3

# Run a single scenario by ID
uv run pytest pipelines/training/automl/autogluon_timeseries_training_pipeline/tests/test_pipeline_functional.py \
    -m functional -v -s --log-cli-level=INFO \
    -k "TC-B-1_timeseries_retail_sales"

# Use the short config for a faster smoke run
AUTOML_FUNCTIONAL_TEST_CONFIG=functional_test_configs_short.json \
uv run pytest pipelines/training/automl/autogluon_timeseries_training_pipeline/tests/test_pipeline_functional.py \
    -m functional -v -s --log-cli-level=INFO

# Write report to a custom location
AUTOML_FUNCTIONAL_TEST_REPORT=/tmp/my_report.json \
uv run pytest pipelines/training/automl/autogluon_timeseries_training_pipeline/tests/test_pipeline_functional.py \
    -m functional -v -s --log-cli-level=INFO
```

> **Note:** Use `-s --log-cli-level=INFO` to see real-time progress and metric
> logs in the terminal. Without these flags, logs are only captured in the
> pytest output on failure.

## Test Flow (per scenario)

1. Read scenario configuration from the JSON config file.
2. Preprocess the dataset CSV (add dummy `item_id` / `timestamp` columns if needed).
3. Upload the preprocessed CSV to S3 at the key specified in `train_data_file_key`.
4. Submit a pipeline run with arguments derived from the config.
5. Wait for the pipeline run to complete (up to `RHOAI_PIPELINE_RUN_TIMEOUT`).
6. Measure wall-clock time for the run.
7. Read `metrics.json` files from S3 artifacts for each refitted model.
8. Validate the `sampled_test_dataset` artifact is present in S3.
9. **Notebook execution** *(optional)*: if `RHOAI_NOTEBOOK_RUNNER_IMAGE` is set,
   download the top-1 model's predictor notebook from S3, run it as a Kubernetes
   Job using `papermill`, and assert the Job succeeds within `RHOAI_NOTEBOOK_RUN_TIMEOUT`.
10. **KServe deployment** *(optional)*: if `RHOAI_DEPLOY_AFTER_TRAINING=true`,
    deploy the top-1 model as a KServe InferenceService, score it with
    `inference_sample`, and assert predictions are returned.
11. Write the scenario result to a per-scenario JSON file (xdist-safe).
12. Assert the run succeeded, completed in under 1 hour, and produced at least
    one model with metrics.

After all scenarios finish, the per-scenario JSON files are merged into
the final `functional_test_report.json`.

## Cleanup / Teardown

After **all** scenarios finish (pass or fail), session-scoped teardown
automatically:

- Deletes uploaded dataset files from S3 (the CSVs uploaded in step 3).
- Deletes all pipeline artifact objects from S3 under each run's prefix
  (`{pipeline-name}/{run-id}/`).

Set `AUTOML_FUNCTIONAL_TEST_KEEP_ARTIFACTS=true` to skip artifact deletion
(uploaded datasets are still cleaned up).

No manual cleanup is required.

## Report Output

The test writes a JSON file to `functional_test_report.json` (or the path
set by `AUTOML_FUNCTIONAL_TEST_REPORT`). The top-level keys are scenario IDs:

```json
{
  "TC-B-1_timeseries_retail_sales": {
    "run_id": "abc123",
    "run_name": "ts-functional-...",
    "started_at": "2026-04-01T12:00:00+00:00",
    "elapsed_seconds": 542.3,
    "elapsed_minutes": 9.04,
    "succeeded": true,
    "target": "Total Amount",
    "id_column": "Product Category",
    "timestamp_column": "Date",
    "known_covariates_names": [],
    "prediction_length": 1,
    "top_n": 3,
    "train_data_file_key": "functional-test/timeseries/retail_sales.csv",
    "dataset_rows": 1000,
    "dataset_features": 8,
    "models": [
      {
        "model_name": "DeepAR_FULL",
        "metrics": {"MASE": 0.85, "MAPE": 0.12},
        "total_predictor_size_bytes": 1048576,
        "total_predictor_size_mb": 1.0,
        "notebook_key": "autogluon-timeseries-training-pipeline/.../DeepAR_FULL/notebooks/automl_predictor_notebook.ipynb"
      }
    ],
    "notebook_run": {
      "succeeded": true,
      "skipped": false,
      "reason": null,
      "error": null,
      "job_name": "nb-deepar-full-abc123",
      "elapsed_seconds": 320.1
    },
    "deployment": {},
    "config": { "..." }
  },
  "TC-B-2_timeseries_traffic": { "..." }
}
```

`notebook_run` is `{}` when `RHOAI_NOTEBOOK_RUNNER_IMAGE` is not set.
`deployment` is `{}` when `RHOAI_DEPLOY_AFTER_TRAINING` is not set.

## Config File Format

The config file is a JSON array. Each entry:

```json
{
  "id": "TC-B-1_timeseries_retail_sales",
  "dataset_path": "data/retail_sales_dataset.csv",
  "target": "Total Amount",
  "id_column": "Product Category",
  "timestamp_column": "Date",
  "known_covariates_names": [],
  "prediction_length": 1,
  "top_n": 3,
  "train_data_file_key": "functional-test/timeseries/retail_sales.csv",
  "tags": ["timeseries", "retail"],
  "add_dummy_item_id": false,
  "add_dummy_timestamp": false,
  "inference_sample": [
    {
      "item_id": ["ts_0", "ts_0", "ts_0"],
      "timestamp": ["2020-01-01", "2020-01-02", "2020-01-03"],
      "Total Amount": [100.0, 120.0, null]
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique scenario identifier (used as pytest ID) |
| `dataset_path` | string | Path to the CSV file relative to `tests/` directory |
| `target` | string | Name of the target column in the CSV |
| `id_column` | string | Name of the item ID column (or the name to use for a dummy column) |
| `timestamp_column` | string | Name of the timestamp column (or the name to use for a dummy column) |
| `known_covariates_names` | list[str] | Names of known covariate columns (can be empty) |
| `prediction_length` | integer | Number of time steps to predict |
| `top_n` | integer | Number of top models to select and refit (1–10) |
| `train_data_file_key` | string | S3 object key where the dataset will be uploaded |
| `tags` | list[str] | Optional tags for documentation/filtering |
| `add_dummy_item_id` | boolean | If true, add a constant `item_id` column before uploading |
| `add_dummy_timestamp` | boolean | If true, add sequential daily dates as a `timestamp` column |
| `inference_sample` | list[dict] | Column-oriented historical rows used for KServe scoring when `RHOAI_DEPLOY_AFTER_TRAINING=true`. Provide at least `prediction_length` rows of history per `item_id`; target values for future steps should be `null`. |

## Registering the `functional` Marker

If pytest warns about unknown markers, register it in `pyproject.toml` or
`pytest.ini`:

```ini
[tool.pytest.ini_options]
markers = [
    "functional: end-to-end functional tests on real RHOAI cluster",
]
```
