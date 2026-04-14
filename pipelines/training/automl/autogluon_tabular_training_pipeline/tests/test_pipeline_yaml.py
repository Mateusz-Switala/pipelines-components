"""Tests verifying committed pipeline.yaml is aligned with the pipeline source."""

from pathlib import Path

from kfp_components.utils.compiled_pipeline_alignment import (
    assert_checked_in_pipeline_yaml_matches_compiled_ir,
)

from ..pipeline import autogluon_tabular_training_pipeline


class TestAutogluonTabularTrainingPipelineYaml:
    """Tests that checked-in compiled YAML matches the tabular training pipeline source."""

    def test_checked_in_pipeline_yaml_matches_source_ir(self):
        """Committed pipeline.yaml matches a fresh compile (IR + deployment; b64 embed + images redacted)."""
        yaml_path = Path(__file__).resolve().parent.parent / "pipeline.yaml"
        assert_checked_in_pipeline_yaml_matches_compiled_ir(
            pipeline_func=autogluon_tabular_training_pipeline,
            checked_in_yaml_path=yaml_path,
        )
