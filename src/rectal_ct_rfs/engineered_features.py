"""Engineered spatial/context features used by the frozen M12 representation."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


STAT_NAMES = ["mean", "std", "min", "max", "p10", "p25", "median", "p75", "p90", "iqr", "mad", "skew", "kurtosis", "entropy", "energy"]
TEXTURE_NAMES = ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy", "neighbor_mean_absdiff", "neighbor_std_absdiff", "neighbor_corr"]


def _safe_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"{prefix}_{name}": 0.0 for name in STAT_NAMES}
    values = values.astype(np.float32, copy=False)
    mean = float(values.mean())
    centered = values - mean
    m2 = float(np.mean(centered ** 2)) + 1e-8
    m3 = float(np.mean(centered ** 3))
    m4 = float(np.mean(centered ** 4))
    hist, _ = np.histogram(values, bins=32)
    probability = hist.astype(np.float64) / max(1, hist.sum())
    probability = probability[probability > 0]
    result = {
        "mean": mean,
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "mad": float(np.median(np.abs(values - np.median(values)))),
        "skew": float(m3 / (m2 ** 1.5)),
        "kurtosis": float(m4 / (m2 ** 2) - 3.0),
        "entropy": float(-(probability * np.log2(probability)).sum()),
        "energy": float(np.mean(values ** 2)),
    }
    return {f"{prefix}_{name}": value for name, value in result.items()}


def _quantized_texture(ct: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float]:
    if int(mask.sum()) < 8:
        return {f"{prefix}_{name}": 0.0 for name in TEXTURE_NAMES}
    values = ct[mask].astype(np.float32)
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        high = low + 1.0
    quantized = np.clip(((ct - low) / (high - low) * 15.0).astype(np.int16), 0, 15)
    pairs = []
    differences = []
    correlations = []
    for axis in range(3):
        valid = np.roll(mask, -1, axis=axis) & mask
        if not valid.any():
            continue
        shifted_q = np.roll(quantized, -1, axis=axis)
        pairs.append((quantized[valid], shifted_q[valid]))
        shifted_ct = np.roll(ct, -1, axis=axis)
        differences.append(np.abs(ct[valid] - shifted_ct[valid]).astype(np.float32))
        x = ct[valid].astype(np.float64)
        y = shifted_ct[valid].astype(np.float64)
        if x.size > 2 and x.std() > 1e-6 and y.std() > 1e-6:
            correlations.append(float(np.corrcoef(x, y)[0, 1]))
    if not pairs:
        return {f"{prefix}_{name}": 0.0 for name in TEXTURE_NAMES}
    i = np.concatenate([pair[0] for pair in pairs])
    j = np.concatenate([pair[1] for pair in pairs])
    glcm = np.zeros((16, 16), dtype=np.float64)
    np.add.at(glcm, (i, j), 1.0)
    np.add.at(glcm, (j, i), 1.0)
    glcm /= max(glcm.sum(), 1.0)
    ii, jj = np.indices(glcm.shape)
    diff = np.concatenate(differences)
    return {
        f"{prefix}_glcm_contrast": float(((ii - jj) ** 2 * glcm).sum()),
        f"{prefix}_glcm_dissimilarity": float((np.abs(ii - jj) * glcm).sum()),
        f"{prefix}_glcm_homogeneity": float((glcm / (1.0 + np.abs(ii - jj))).sum()),
        f"{prefix}_glcm_energy": float((glcm ** 2).sum()),
        f"{prefix}_neighbor_mean_absdiff": float(diff.mean()),
        f"{prefix}_neighbor_std_absdiff": float(diff.std()),
        f"{prefix}_neighbor_corr": float(np.mean(correlations)) if correlations else 0.0,
    }


def extract_engineered_features(ct: np.ndarray, mask: np.ndarray, spacing_mm: float = 2.5) -> dict[str, float]:
    """Return the 46 engineered M12 inputs from a preprocessed CT/mask pair.

    The arrays must be same-shape 3D volumes. Labels are 1=bowel wall,
    2=primary tumor, and 3=regional lymph nodes. The frozen analysis used
    isotropic 2.5-mm cached arrays for this engineered block.
    """
    if ct.shape != mask.shape or ct.ndim != 3:
        raise ValueError("ct and mask must be same-shape 3D arrays")
    wall, tumor, nodes = mask == 1, mask == 2, mask == 3
    union = wall | tumor | nodes
    result: dict[str, float] = {}
    if tumor.any():
        distance_wall = ndi.distance_transform_edt(~wall, sampling=spacing_mm)
        distance_nodes = ndi.distance_transform_edt(~nodes, sampling=spacing_mm)
        result["tumor_wall_distance_mean"] = float(distance_wall[tumor].mean())
        result["tumor_wall_distance_min"] = float(distance_wall[tumor].min())
        result["tumor_nodes_distance_mean"] = float(distance_nodes[tumor].mean())
        result["tumor_nodes_distance_min"] = float(distance_nodes[tumor].min())
        dilated = ndi.binary_dilation(tumor, iterations=1)
        result["tumor_wall_contact_voxels"] = float(np.count_nonzero(dilated & wall))
        result["tumor_nodes_contact_voxels"] = float(np.count_nonzero(dilated & nodes))
    else:
        for name in ("tumor_wall_distance_mean", "tumor_wall_distance_min", "tumor_nodes_distance_mean", "tumor_nodes_distance_min", "tumor_wall_contact_voxels", "tumor_nodes_contact_voxels"):
            result[name] = 0.0
    result["wall_nodes_contact_voxels"] = float(np.count_nonzero(ndi.binary_dilation(wall, iterations=1) & nodes))
    result["union_volume_mm3"] = float(union.sum() * spacing_mm ** 3)
    result["union_fraction"] = float(union.mean())
    result.update(_safe_stats(ct[union], "union"))
    shell = ndi.binary_dilation(tumor, iterations=4) & ~tumor
    result.update(_safe_stats(ct[shell], "peritumor_shell"))
    result.update(_quantized_texture(ct, shell, "peritumor_shell"))
    return {f"eng__{name}": value for name, value in result.items()}
