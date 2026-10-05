"""Maintainer utility: export a patient-data-free model from the private fit artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np


HORIZONS = np.asarray([12.0, 24.0, 36.0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-model", type=Path, required=True)
    parser.add_argument("--private-module-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.private_module_dir.resolve()))
    private = joblib.load(args.private_model)
    transformers = private["transformers"]
    full = transformers["full"]
    cnn = transformers["cnn"]
    feature_groups = private["feature_groups"]
    radiomics_features = feature_groups["full_rad"] + feature_groups["engineered_cross"]
    cnn_features = feature_groups["cnn"]

    public_preprocess = {
        "clinical_pipe": transformers["clinical"].pipe,
        "radiomics_imputer": full.imputer,
        "radiomics_keep": full.keep,
        "radiomics_scaler": full.scaler,
        "radiomics_pca": full.pca,
        "cnn_imputer": cnn.imputer,
        "cnn_keep": cnn.keep,
        "cnn_scaler": cnn.scaler,
        "cnn_pca": cnn.pca,
        "final_keep": transformers["_final_keep"],
        "radiomics_features": radiomics_features,
        "cnn_features": cnn_features,
    }

    cox = private["cox_model"]._model
    timeline = cox.baseline_cumulative_hazard_.index.to_numpy(float)
    baseline = cox.baseline_cumulative_hazard_.iloc[:, 0].to_numpy(float)
    baseline_at_horizons = np.interp(HORIZONS, timeline, baseline)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(public_preprocess, args.output_dir / "m12_preprocessing.joblib", compress=3)
    np.savez_compressed(
        args.output_dir / "m12_cox_parameters.npz",
        beta=cox.params_.to_numpy(float),
        norm_mean=cox._norm_mean.to_numpy(float),
        horizons_months=HORIZONS,
        baseline_cumulative_hazard=baseline_at_horizons,
    )
    metadata = {
        "analysis_id": private["analysis_id"],
        "model": "M12",
        "endpoint": "recurrence-free survival",
        "event_definition": "first local recurrence, distant metastasis, or death from any cause",
        "time_origin": "curative-intent surgery",
        "horizons_months": HORIZONS.tolist(),
        "n_final_features": len(private["feature_names"]),
        "feature_names": private["feature_names"],
        "removed_predictors": private["removed_predictors"],
        "privacy_note": "This export omits patient identifiers, individual follow-up times, outcomes, and patient-level predictions.",
        "intended_use": "Research use only; not a medical device and not for direct clinical decision-making.",
    }
    (args.output_dir / "m12_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
