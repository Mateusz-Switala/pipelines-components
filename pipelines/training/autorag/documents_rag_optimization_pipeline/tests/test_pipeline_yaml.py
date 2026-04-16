"""Tests verifying committed pipeline.yaml is aligned with the pipeline source."""

import os
from pathlib import Path

from kfp_components.utils.compiled_pipeline_alignment import (
    assert_checked_in_pipeline_yaml_matches_compiled_ir,
)

_yaml_path = Path(__file__).resolve().parent.parent / "pipeline.yaml"

os.environ.setdefault(
    "RELATED_IMAGE_MPI_AUTORAG_RUNTIME",
    "registry.redhat.io/rhoai/odh-autorag-rhel9@sha256:152c98afbf558367807d9dc0bd0a203a71936476039c7130eeecc079d0605cf2",
)

from ..pipeline import documents_rag_optimization_pipeline  # noqa: E402


class TestDocumentsRagOptimizationPipelineYaml:
    """Tests that checked-in compiled YAML matches the documents RAG optimization pipeline source."""

    def test_checked_in_pipeline_yaml_matches_source_ir(self):
        """Committed pipeline.yaml matches a fresh compile (IR + deployment; b64 embed + images redacted)."""
        assert_checked_in_pipeline_yaml_matches_compiled_ir(
            pipeline_func=documents_rag_optimization_pipeline,
            checked_in_yaml_path=_yaml_path,
        )
