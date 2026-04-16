"""Tests verifying committed pipeline.yaml is aligned with the pipeline source."""

import os
from pathlib import Path

from kfp_components.utils.compiled_pipeline_alignment import (
    assert_checked_in_pipeline_yaml_matches_compiled_ir,
)

_yaml_path = Path(__file__).resolve().parent.parent / "pipeline.yaml"

os.environ.setdefault(
    "RELATED_IMAGE_MPI_AUTOML_RUNTIME",
    "registry.redhat.io/rhoai/odh-automl-rhel9@sha256:6d4da6c8201577db131f37d6a8572b13b6c1d01a64115b6685ffe8e053f5fe79",
)

from ..pipeline import autogluon_tabular_training_pipeline  # noqa: E402


class TestAutogluonTabularTrainingPipelineYaml:
    """Tests that checked-in compiled YAML matches the tabular training pipeline source."""

    def test_checked_in_pipeline_yaml_matches_source_ir(self):
        """Committed pipeline.yaml matches a fresh compile (IR + deployment; b64 embed + images redacted)."""
        assert_checked_in_pipeline_yaml_matches_compiled_ir(
            pipeline_func=autogluon_tabular_training_pipeline,
            checked_in_yaml_path=_yaml_path,
        )
