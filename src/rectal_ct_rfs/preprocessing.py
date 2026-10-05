"""Privacy-preserving preprocessing for the frozen M12 model."""

from __future__ import annotations

import numpy as np
import pandas as pd


CONTINUOUS = [
    "age",
    "tumor_size_cm",
    "distance_from_anal_verge_cm",
    "examined_node_count",
    "cea",
    "ca19_9",
]

CATEGORICAL = [
    "sex",
    "differentiation",
    "margin_status",
    "pni",
    "lvi",
    "pt_stage",
    "pn_stage",
]


def _map_binary(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    mapping = {
        "yes": 1.0,
        "present": 1.0,
        "positive": 1.0,
        "true": 1.0,
        "no": 0.0,
        "absent": 0.0,
        "negative": 0.0,
        "false": 0.0,
        "是": 1.0,
        "有": 1.0,
        "阳性": 1.0,
        "否": 0.0,
        "无": 0.0,
        "阴性": 0.0,
    }
    mapped = text.str.lower().map(mapping)
    numeric = pd.to_numeric(values, errors="coerce")
    return mapped.where(mapped.notna(), numeric)


def clean_clinical(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the same deterministic clinical cleaning used for model fitting."""
    missing = sorted(set(CONTINUOUS + CATEGORICAL) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing clinical columns: {missing}")
    out = frame.copy()
    for name in CONTINUOUS:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    for name in ("cea", "ca19_9"):
        out[name] = np.log1p(np.clip(out[name], 0, None))
    for name in ("margin_status", "pni", "lvi"):
        out[name] = _map_binary(out[name]).astype("Int64").astype(str)
    for name in CATEGORICAL:
        out[name] = out[name].astype(str).str.strip()
    return out


def transform_inputs(frame: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Transform clinical, radiomic-spatial, and CNN features into the M12 design."""
    required = bundle["radiomics_features"] + bundle["cnn_features"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        preview = missing[:10]
        suffix = " ..." if len(missing) > 10 else ""
        raise ValueError(f"Missing imaging features ({len(missing)}): {preview}{suffix}")

    clinical = bundle["clinical_pipe"].transform(clean_clinical(frame))

    radiomics = frame[bundle["radiomics_features"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    radiomics = bundle["radiomics_imputer"].transform(radiomics)
    radiomics = bundle["radiomics_scaler"].transform(radiomics[:, bundle["radiomics_keep"]])
    radiomics = bundle["radiomics_pca"].transform(radiomics)

    cnn = frame[bundle["cnn_features"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    cnn = bundle["cnn_imputer"].transform(cnn)
    cnn = bundle["cnn_scaler"].transform(cnn[:, bundle["cnn_keep"]])
    cnn = bundle["cnn_pca"].transform(cnn)

    design = np.nan_to_num(
        np.column_stack([clinical, radiomics, cnn]),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return design[:, bundle["final_keep"]]
