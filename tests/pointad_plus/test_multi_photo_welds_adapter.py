"""Tests for the multi-photometric welds adapter."""
from pathlib import Path
import json
import pytest

from src.pointad_plus import multi_photo_welds_adapter as adp


def test_eyecandies_weld_root_exists():
    p = Path("F:/dataset/LUT_AD_DataSet/Dataset_3D/Eyecandies_Weld/weld")
    assert p.is_dir()
    # confirms the 5 modality subdirs exist on at least one sample
    sub = next((p / "train" / "good").iterdir())  # first modality subdir
    assert sub.name in {"lut", "phong", "diffuse", "specular", "normal", "depth", "gt"}


def test_sample_has_five_modality_paths(tmp_path):
    out = tmp_path / "all_meta.json"
    adp.emit_manifest(
        source_root=Path("F:/dataset/LUT_AD_DataSet/Dataset_3D/Eyecandies_Weld/weld"),
        out_path=out,
    )
    meta = json.loads(out.read_text())
    assert "train" in meta and "test" in meta
    train_weld = meta["train"]["weld"]
    assert len(train_weld) > 100
    s = train_weld[0]
    for k in ("lut", "phong", "diffuse", "specular", "normal"):
        assert k in s["multi_photo"]
        assert Path(s["multi_photo"][k]).exists()


def test_test_split_has_anomalous(tmp_path):
    out = tmp_path / "all_meta.json"
    adp.emit_manifest(
        source_root=Path("F:/dataset/LUT_AD_DataSet/Dataset_3D/Eyecandies_Weld/weld"),
        out_path=out,
    )
    meta = json.loads(out.read_text())
    test_weld = meta["test"]["weld"]
    anom = [s for s in test_weld if s["anomaly"]]
    assert len(anom) == 541  # matches Phase 1
    classes = {s["specie_name"] for s in anom}
    assert {"pseudo_soldering", "pinhole", "pit", "burst",
            "fish_scale_welding", "bump", "combined"}.issubset(classes)
