"""Tests verifying committed pipeline.yaml is aligned with the pipeline source."""

import os
from pathlib import Path

from kfp_components.utils.compiled_pipeline_alignment import (
    assert_checked_in_pipeline_yaml_matches_compiled_ir,
)

_yaml_path = Path(__file__).resolve().parent.parent / "pipeline.yaml"


def _set_automl_image_from_yaml(yaml_path: Path) -> None:
    """Set RELATED_IMAGE_MPI_AUTOML_RUNTIME from the committed pipeline.yaml if not already set."""
    if os.environ.get("RELATED_IMAGE_MPI_AUTOML_RUNTIME") or not yaml_path.is_file():
        return
    import yaml

    with yaml_path.open() as f:
        for doc in yaml.safe_load_all(f):
            if isinstance(doc, dict) and "deploymentSpec" in doc:
                for executor in doc["deploymentSpec"].get("executors", {}).values():
                    img = executor.get("container", {}).get("image")
                    if img:
                        os.environ["RELATED_IMAGE_MPI_AUTOML_RUNTIME"] = img
                        return


_set_automl_image_from_yaml(_yaml_path)

from ..pipeline import autogluon_timeseries_training_pipeline  # noqa: E402


class TestAutogluonTimeseriesTrainingPipelineYaml:
    """Tests that checked-in compiled YAML matches the timeseries training pipeline source."""

    def test_checked_in_pipeline_yaml_matches_source_ir(self):
        """Committed pipeline.yaml matches a fresh compile (IR + deployment; b64 embed + images redacted)."""
        assert_checked_in_pipeline_yaml_matches_compiled_ir(
            pipeline_func=autogluon_timeseries_training_pipeline,
            checked_in_yaml_path=_yaml_path,
        )
