#!/usr/bin/env python3
"""Minimal helper: create only a KServe InferenceService CR (nothing else).

Does **not** create ServingRuntime or S3 secrets. ``storage.key`` must match a key in the
namespace ``storage-config`` Secret (e.g. ``connection-to-minio-autogluon-artifacts``).

**From repository root**::

    uv run python pipelines/training/automl/autogluon_tabular_training_pipeline/tests/create_inference_service_only.py

Requires ``.env`` with ``RHOAI_URL``, ``RHOAI_TOKEN``, ``RHOAI_PROJECT_NAME``, S3 vars
(see ``integration_config.py``), ``RHOAI_DEBUG_STORAGE_URI``, and ``RHOAI_ISVC_STORAGE_KEY`` or
``RHOAI_TEST_S3_SECRET_NAME``. Set ``RHOAI_SERVING_RUNTIME_NAME`` to the **ServingRuntime**
``metadata.name`` in your project (e.g. ``tabulat-production-image-v2``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


def _env_or_default(key: str, default: str) -> str:
    """Treat empty ``KEY=`` in ``.env`` as unset (``python-dotenv`` leaves the key present)."""
    return (os.environ.get(key) or "").strip() or default


def _dump_inference_service(co, namespace: str, name: str) -> dict | None:
    """Fetch ISVC and log spec/status; return object or None."""
    from kubernetes.client.rest import ApiException

    from pipelines.training.automl.autogluon_tabular_training_pipeline.tests import (
        test_pipeline_functional as tpf,
    )

    try:
        isvc = co.get_namespaced_custom_object(
            group=tpf._KSERVE_GROUP,
            version=tpf._KSERVE_ISVC_VERSION,
            namespace=namespace,
            plural=tpf._KSERVE_ISVC_PLURAL,
            name=name,
        )
    except ApiException as e:
        logger.error("get InferenceService %s/%s failed: %s", namespace, name, e)
        return None
    logger.info("spec (as stored on cluster):\n%s", json.dumps(isvc.get("spec"), indent=2, default=str))
    logger.info("status:\n%s", json.dumps(isvc.get("status"), indent=2, default=str))
    return isvc


def main() -> int:
    from kubernetes import client
    from kubernetes.client.rest import ApiException

    from pipelines.training.automl.autogluon_tabular_training_pipeline.tests import integration_config
    from pipelines.training.automl.autogluon_tabular_training_pipeline.tests.conftest import (
        _build_temp_kubeconfig,
    )
    from pipelines.training.automl.autogluon_tabular_training_pipeline.tests import (
        test_pipeline_functional as tpf,
    )

    integration_config._ensure_dotenv_loaded()

    p = argparse.ArgumentParser(description="Create a KServe InferenceService only (no runtime/secret).")
    p.add_argument(
        "--storage-uri",
        default=os.environ.get("RHOAI_DEBUG_STORAGE_URI", ""),
        help="s3://.../predictor (or RHOAI_DEBUG_STORAGE_URI)",
    )
    p.add_argument(
        "--storage-key",
        default=os.environ.get("RHOAI_ISVC_STORAGE_KEY", ""),
        help="storage-config key — env RHOAI_ISVC_STORAGE_KEY or RHOAI_TEST_S3_SECRET_NAME fallback",
    )
    p.add_argument(
        "--inference-service",
        default="",
        help="Kubernetes ISVC name (default: automl-isvc-only-<pid>)",
    )
    p.add_argument(
        "--runtime",
        default=_env_or_default("RHOAI_SERVING_RUNTIME_NAME", "kserve-autogluonserver"),
        help="ServingRuntime metadata.name (RHOAI_SERVING_RUNTIME_NAME)",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    cfg = integration_config.get_rhoai_config()
    if not cfg:
        logger.error("Missing integration config — see integration_config.py and .env")
        return 1

    storage_uri = (args.storage_uri or "").strip()
    if not storage_uri:
        logger.error("Set --storage-uri or RHOAI_DEBUG_STORAGE_URI")
        return 1

    storage_key = (args.storage_key or "").strip() or (cfg.get("s3_secret_name") or "").strip()
    if not storage_key:
        logger.error("Set --storage-key, RHOAI_ISVC_STORAGE_KEY, or RHOAI_TEST_S3_SECRET_NAME")
        return 1

    namespace = cfg["rhoai_project"]
    isvc_name = (args.inference_service or "").strip() or f"automl-isvc-only-{os.getpid():x}"[:57]
    runtime_name = (args.runtime or "").strip() or _env_or_default(
        "RHOAI_SERVING_RUNTIME_NAME",
        "kserve-autogluonserver",
    )
    if not runtime_name.strip():
        logger.error("Serving runtime name is empty — set RHOAI_SERVING_RUNTIME_NAME or --runtime")
        return 1

    kube_path = _build_temp_kubeconfig(cfg["rhoai_url"], cfg["rhoai_token"], namespace)
    try:
        tpf._load_k8s_config(kube_path)
        co = client.CustomObjectsApi()
        logger.info(
            "Creating InferenceService %r in %r (runtime=%r storage.key=%r)",
            isvc_name,
            namespace,
            runtime_name,
            storage_key,
        )
        try:
            tpf._create_inference_service(co, namespace, isvc_name, runtime_name, storage_uri, storage_key)
        except ApiException as e:
            logger.error(
                "Kubernetes rejected InferenceService (HTTP %s): %s — %s",
                e.status,
                e.reason,
                getattr(e, "body", "") or e,
            )
            return 1

        obj = _dump_inference_service(co, namespace, isvc_name)
        if not obj:
            return 1
        logger.info("Done. oc get inferenceservice %s -n %s -o yaml", isvc_name, namespace)
        return 0
    finally:
        try:
            Path(kube_path).unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
