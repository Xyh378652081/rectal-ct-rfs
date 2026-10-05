"""Core multistructure radiomic and spatial feature definitions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor


LABELS = {1: "wall", 2: "tumor", 3: "nodes"}
TARGET_SPACING = (1.0, 1.0, 1.0)


def resample_pair(image: sitk.Image, mask: sitk.Image):
    old_spacing = image.GetSpacing()
    old_size = image.GetSize()
    new_size = [max(1, int(round(old_size[i] * old_spacing[i] / TARGET_SPACING[i]))) for i in range(3)]
    image_r = sitk.Resample(
        image, new_size, sitk.Transform(), sitk.sitkBSpline,
        image.GetOrigin(), TARGET_SPACING, image.GetDirection(), -1024.0, sitk.sitkFloat32,
    )
    mask_r = sitk.Resample(
        mask, new_size, sitk.Transform(), sitk.sitkNearestNeighbor,
        mask.GetOrigin(), TARGET_SPACING, mask.GetDirection(), 0, sitk.sitkUInt8,
    )
    return image_r, mask_r


def _geometry(mask: sitk.Image, label: int):
    binary = sitk.Cast(mask == label, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(binary)
    if not stats.HasLabel(1):
        return binary, None
    return binary, {
        "volume_ml": float(stats.GetPhysicalSize(1) / 1000.0),
        "centroid": np.asarray(stats.GetCentroid(1), dtype=float),
    }


def extract_case(image_path: str | Path, mask_path: str | Path, parameter_file: str | Path) -> dict[str, float]:
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    if image.GetDimension() != 3 or mask.GetDimension() != 3:
        raise ValueError("This public extractor expects 3D image and mask volumes")
    if image.GetSize() != mask.GetSize():
        raise ValueError("Image and mask geometry do not match")
    image, mask = resample_pair(image, mask)
    extractor = featureextractor.RadiomicsFeatureExtractor(str(parameter_file))
    features: dict[str, float] = {}
    binary = {}
    geometry = {}
    for label, name in LABELS.items():
        binary[name], geometry[name] = _geometry(mask, label)
        if geometry[name] is None:
            continue
        result = extractor.execute(image, binary[name], label=1)
        for key, value in result.items():
            if key.startswith("original_"):
                features[f"{name}__{key}"] = float(value)

    wall, tumor, nodes = geometry["wall"], geometry["tumor"], geometry["nodes"]
    if wall is not None and tumor is not None:
        features["spatial_wall_tumor_centroid_distance_mm"] = float(np.linalg.norm(wall["centroid"] - tumor["centroid"]))
        features["spatial_tumor_wall_volume_ratio"] = tumor["volume_ml"] / max(wall["volume_ml"], 1e-9)
    if tumor is not None and nodes is not None:
        distance = sitk.SignedMaurerDistanceMap(binary["tumor"], insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
        values = sitk.GetArrayViewFromImage(distance)[sitk.GetArrayViewFromImage(binary["nodes"]) > 0]
        values = np.maximum(values.astype(float), 0.0)
        features["spatial_tumor_nodes_centroid_distance_mm"] = float(np.linalg.norm(tumor["centroid"] - nodes["centroid"]))
        features["spatial_tumor_nodes_min_distance_mm"] = float(values.min())
        features["spatial_nodes_within_5mm_fraction"] = float(np.mean(values <= 5.0))
        features["spatial_nodes_within_10mm_fraction"] = float(np.mean(values <= 10.0))
        features["spatial_nodes_tumor_volume_ratio"] = nodes["volume_ml"] / max(tumor["volume_ml"], 1e-9)
    return features
