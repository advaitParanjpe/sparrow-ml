import pytest
from sparrowml.artifacts.schemas import DeploymentManifest, ModelMetadata

def test_manifest_accepts_relative_paths():
    model = ModelMetadata("m", (1, 4), 2, "int8")
    manifest = DeploymentManifest(model, "a/input.bin", "a/dense.bin", "a/sparse.bin", "a/meta.bin", "a/bias.bin", "a/out.bin", "a/program.bin", "a/data.bin")
    assert manifest.program_image_path == "a/program.bin"

def test_manifest_rejects_absolute_path():
    model = ModelMetadata("m", (1,), 1, "int8")
    with pytest.raises(ValueError): DeploymentManifest(model, "/bad", "a", "a", "a", "a", "a", "a", "a")

def test_metadata_rejects_missing_fields():
    with pytest.raises(ValueError): ModelMetadata("", (), 0, "int8")
