"""Inference utilities for the frozen privacy-preserving M12 model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .preprocessing import transform_inputs


class M12Model:
    """Load the frozen preprocessing bundle and Cox model parameters."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.preprocess = joblib.load(model_dir / "m12_preprocessing.joblib")
        arrays = np.load(model_dir / "m12_cox_parameters.npz")
        self.beta = arrays["beta"]
        self.norm_mean = arrays["norm_mean"]
        self.horizons = arrays["horizons_months"]
        self.baseline_cumulative_hazard = arrays["baseline_cumulative_hazard"]
        self.metadata = json.loads((model_dir / "m12_metadata.json").read_text(encoding="utf-8"))

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return relative risk and absolute event risk at 12, 24, and 36 months."""
        design = transform_inputs(frame, self.preprocess)
        log_partial_hazard = (design - self.norm_mean) @ self.beta
        relative_risk = np.exp(log_partial_hazard)
        event_risk = 1.0 - np.exp(
            -relative_risk[:, None] * self.baseline_cumulative_hazard[None, :]
        )
        result = pd.DataFrame({"relative_risk": relative_risk}, index=frame.index)
        for idx, horizon in enumerate(self.horizons):
            result[f"event_risk_{int(horizon)}m"] = event_risk[:, idx]
        return result
