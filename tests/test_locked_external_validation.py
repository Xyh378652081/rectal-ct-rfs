from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "locked_external_validation", ROOT / "analysis" / "locked_external_validation.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_predictions():
    rows = []
    for center_index, center in enumerate(("Center 2", "Center 3", "Center 4")):
        for patient in range(12):
            event = int(patient % 4 == 0)
            time = 8.0 + patient * 4.0 + center_index
            score = patient / 12.0 + center_index * 0.03
            rows.append({
                "patient_id": f"synthetic-{center_index}-{patient}",
                "center": center,
                "model": "M12",
                "time_months": time,
                "event": event,
                "risk_score": score,
                "risk12": min(0.95, 0.05 + score * 0.25),
                "risk24": min(0.95, 0.08 + score * 0.35),
                "risk36": min(0.95, 0.10 + score * 0.45),
            })
    return pd.DataFrame(rows)


def test_locked_metrics_are_finite():
    frame = synthetic_predictions()
    result = MODULE.metrics(frame)
    assert np.isfinite(result["auc"]["36"])
    assert np.isfinite(result["brier"]["36"])
    assert np.isfinite(result["calibration36"]["intercept"])


def test_center_stratified_bootstrap_preserves_center_sizes():
    frame = synthetic_predictions()
    index = MODULE.draw_center_stratified(frame, np.random.default_rng(20261004))
    sampled = frame.iloc[index]
    assert sampled.center.value_counts().sort_index().equals(
        frame.center.value_counts().sort_index()
    )
