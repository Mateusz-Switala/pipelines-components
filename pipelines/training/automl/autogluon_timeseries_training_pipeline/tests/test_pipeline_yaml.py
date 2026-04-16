"""Tests verifying committed pipeline.yaml is aligned with the pipeline source."""

import os
from pathlib import Path

from kfp_components.utils.compiled_pipeline_alignment import (
    assert_checked_in_pipeline_yaml_matches_compiled_ir,
)

_yaml_path = Path(__file__).resolve().parent.parent / "pipeline.yaml"

os.environ.setdefault(
    "RELATED_IMAGE_MPI_AUTOML_RUNTIME",
    "registry.redhat.io/rhoai/odh-automl-rhel9@sha256:d943beee403c071e18b939206aa6f6284135644070181bb8d350baf056cc0564",
)

from ..pipeline import autogluon_timeseries_training_pipeline  # noqa: E402


class TestAutogluonTimeseriesTrainingPipelineYaml:
    """Tests that checked-in compiled YAML matches the timeseries training pipeline source."""

    def test_checked_in_pipeline_yaml_matches_source_ir(self):
        """Committed pipeline.yaml matches a fresh compile (IR + deployment; b64 embed + images redacted)."""
        assert_checked_in_pipeline_yaml_matches_compiled_ir(
            pipeline_func=autogluon_timeseries_training_pipeline,
            checked_in_yaml_path=_yaml_path,
        )
