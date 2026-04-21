# Functional Tests — AutoGluon Tabular Training Pipeline (TC-A)

Functional tests for the `autogluon_tabular_training_pipeline`. Each scenario runs the
full pipeline end-to-end on a real RHOAI cluster with real datasets.

## Scenarios

| ID | Dataset | Task Type | top_n | Label Column | S3 Key Shape |
|----|---------|-----------|-------|--------------|--------------|
| TC-A-1 | House Price Prediction (2,000 rows, 9 features) | regression | 1 | Price | shallow |
| TC-A-2 | ChurnModelling (10,000 rows, 13 features) | binary | 5 | Exited | nested path |
| TC-A-3 | Covertype (581,012 rows, 54 features) | multiclass | 3 | Cover_Type | shallow |

Scenarios are defined in `functional_test_configs.json`. You can point to a
custom config file via the `AUTOML_FUNCTIONAL_TEST_CONFIG` environment variable.

## Prerequisites

1. **RHOAI cluster** with Data Science Pipelines enabled and a
   DataSciencePipelinesApplication (DSPA) deployed.
2. **S3-compatible storage** (MinIO, AWS S3, etc.) accessible from the cluster
   and from the machine running the tests.
3. **Python environment** with test dependencies installed:

   ```bash
   pip install pytest boto3 kfp python-dotenv pyyaml kubernetes
   # or, from the repo root:
   uv sync --extra test
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
   successfully.

   | Variable | Required | Description |
   |----------|----------|-------------|
   | `RHOAI_NOTEBOOK_RUNNER_IMAGE` | no | Container image with `papermill` and `autogluon` for notebook execution. When unset the step is skipped. |
   | `RHOAI_NOTEBOOK_RUN_TIMEOUT` | no | Seconds to wait for the notebook Job to finish (default: `600`) |
   | `RHOAI_KSERVE_CA_BUNDLE_CONFIGMAP` | no | Name of the ConfigMap with CA bundle for the KServe storage initializer. Required when S3 uses a custom/self-signed TLS certificate (e.g. MinIO on OpenShift). |

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
   - `House Price Prediction Dataset.csv` — Housing price regression
   - `ChurnModelling.csv` — Customer churn binary classification
   - `covtype.csv` — Covertype multiclass classification

## Running the Tests

```bash
# Run all functional scenarios (sequential)
uv run pytest pipelines/training/automl/autogluon_tabular_training_pipeline/tests/test_pipeline_functional.py \
    -m functional -v -s --log-cli-level=INFO

# Run all scenarios in parallel (3 workers via pytest-xdist)
uv run pytest pipelines/training/automl/autogluon_tabular_training_pipeline/tests/test_pipeline_functional.py \
    -m functional -v -s --log-cli-level=INFO -n 3

# Run a single scenario by ID
uv run pytest pipelines/training/automl/autogluon_tabular_training_pipeline/tests/test_pipeline_functional.py \
    -m functional -v -s --log-cli-level=INFO \
    -k "TC-A-1_regression_housing"

# Use the short config for a faster smoke run
AUTOML_FUNCTIONAL_TEST_CONFIG=functional_test_configs_short.json \
uv run pytest pipelines/training/automl/autogluon_tabular_training_pipeline/tests/test_pipeline_functional.py \
    -m functional -v -s --log-cli-level=INFO

# Write report to a custom location
AUTOML_FUNCTIONAL_TEST_REPORT=/tmp/my_report.json \
uv run pytest pipelines/training/automl/autogluon_tabular_training_pipeline/tests/test_pipeline_functional.py \
    -m functional -v -s --log-cli-level=INFO
```

> **Note:** Use `-s --log-cli-level=INFO` to see real-time progress and metric
> logs in the terminal. Without these flags, logs are only captured in the
> pytest output on failure.

## Test Flow (per scenario)

1. Read scenario configuration from the JSON config file.
2. Upload the dataset CSV to S3 at the key specified in `train_data_file_key`.
3. Submit a pipeline run with arguments derived from the config.
4. Wait for the pipeline run to complete (up to `RHOAI_PIPELINE_RUN_TIMEOUT`).
5. Measure wall-clock time for the run.
6. Read `metrics.json` files from S3 artifacts for each refitted model.
7. **Notebook execution** *(optional)*: if `RHOAI_NOTEBOOK_RUNNER_IMAGE` is set,
   download the top-1 model's predictor notebook from S3, run it as a Kubernetes
   Job using `papermill`, and assert the Job succeeds within `RHOAI_NOTEBOOK_RUN_TIMEOUT`.
8. **KServe deployment** *(optional)*: if `RHOAI_DEPLOY_AFTER_TRAINING=true`,
   deploy the top-1 model as a KServe InferenceService, score it with
   `inference_sample`, and assert predictions are returned.
9. Write the scenario result to a per-scenario JSON file (xdist-safe).
10. Assert the run succeeded, completed in under 1 hour, and produced at least
    one model with metrics.

After all scenarios finish, the per-scenario JSON files are merged into
the final `functional_test_report.json`.

## Cleanup / Teardown

After **all** scenarios finish (pass or fail), session-scoped teardown
automatically:

- Deletes uploaded dataset files from S3 (the CSVs uploaded in step 2).
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
  "TC-A-1_regression_housing": {
    "run_id": "abc123",
    "run_name": "automl-functional-...",
    "started_at": "2026-04-01T12:00:00+00:00",
    "elapsed_seconds": 542.3,
    "elapsed_minutes": 9.04,
    "succeeded": true,
    "task_type": "regression",
    "top_n": 1,
    "label_column": "Price",
    "train_data_file_key": "functional-test/housing_prices.csv",
    "models": [
      {
        "model_name": "LightGBM_BAG_L1_FULL",
        "metrics": {"r2": 0.92, "root_mean_squared_error": -15234.5},
        "total_predictor_size_bytes": 1048576,
        "total_predictor_size_mb": 1.0,
        "notebook_key": "autogluon-tabular-training-pipeline/.../LightGBM_BAG_L1_FULL/notebooks/automl_predictor_notebook.ipynb"
      }
    ],
    "notebook_run": {
      "succeeded": true,
      "skipped": false,
      "reason": null,
      "error": null,
      "job_name": "nb-lightgbm-bag-l1-full-abc123",
      "elapsed_seconds": 180.5
    },
    "deployment": {},
    "config": { "..." }
  },
  "TC-A-2_binary_churn": { "..." }
}
```

`notebook_run` is `{}` when `RHOAI_NOTEBOOK_RUNNER_IMAGE` is not set.
`deployment` is `{}` when `RHOAI_DEPLOY_AFTER_TRAINING` is not set.

## Config File Format

The config file is a JSON array. Each entry:

```json
{
  "id": "TC-A-1_regression_housing",
  "dataset_path": "data/House Price Prediction Dataset.csv",
  "label_column": "Price",
  "task_type": "regression",
  "top_n": 1,
  "train_data_file_key": "functional-test/housing_prices.csv",
  "tags": ["regression", "smoke"],
  "inference_sample": [
    {
      "Id": [1],
      "Area": [1360],
      "Bedrooms": [5],
      "Bathrooms": [4],
      "Floors": [3],
      "YearBuilt": [1970],
      "Location": ["Downtown"],
      "Condition": ["Excellent"],
      "Garage": ["No"]
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique scenario identifier (used as pytest ID) |
| `dataset_path` | string | Path to the CSV file relative to `tests/` directory |
| `label_column` | string | Name of the target column in the CSV |
| `task_type` | string | One of `regression`, `binary`, `multiclass` |
| `top_n` | integer | Number of top models to select and refit (1–10) |
| `train_data_file_key` | string | S3 object key where the dataset will be uploaded |
| `tags` | list[str] | Optional tags for documentation/filtering |
| `inference_sample` | list[dict] | Column-oriented sample rows used for KServe scoring when `RHOAI_DEPLOY_AFTER_TRAINING=true`. Each dict maps column names to lists of values. The label column must be omitted. |

> **Note on `inference_sample`:** include all feature columns the model was trained
> on, including any ID columns present in the training CSV (e.g. `RowNumber`,
> `CustomerId` for the churn dataset). Omitting a column the model requires will
> cause the InferenceService to return a 500 error.

## Registering the `functional` Marker

If pytest warns about unknown markers, register it in `pyproject.toml` or
`pytest.ini`:

```ini
[tool.pytest.ini_options]
markers = [
    "functional: end-to-end functional tests on real RHOAI cluster",
]
```
