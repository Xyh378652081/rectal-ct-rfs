from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rectal_ct_rfs import M12Model
from rectal_ct_rfs.preprocessing import transform_inputs


REMOVED = {"regional_node_metastasis", "adjacent_organ_invasion", "metastatic_node_count"}


def test_synthetic_prediction_is_finite():
    frame = pd.read_csv(ROOT / "examples" / "synthetic_input.csv")
    model = M12Model(ROOT / "models")
    result = model.predict(frame)
    assert result.shape == (1, 4)
    assert np.isfinite(result.to_numpy()).all()
    assert ((result.filter(like="event_risk") >= 0) & (result.filter(like="event_risk") <= 1)).all().all()
    assert transform_inputs(frame, model.preprocess).shape == (1, 55)
    assert REMOVED.isdisjoint(frame.columns)
    assert model.metadata["analysis_id"] == "RECTAL_LATEST_CLINICAL_V3_20261001"
