"""Locked external-validation metrics used in the final manuscript.

The input is a patient-level prediction table created inside an approved secure
environment. Patient-level inputs and outputs must not be committed to version
control. Required columns are: patient_id, center, model, time_months, event,
risk_score, risk12, risk24, and risk36.

For external validation, the censoring distribution is estimated separately
within each center. It is re-estimated inside every center-stratified patient
bootstrap sample. Cases use G(t-) and controls use G(horizon-).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

HORIZONS = (12, 24, 36)
REQUIRED_COLUMNS = {
    "patient_id", "center", "model", "time_months", "event", "risk_score",
    "risk12", "risk24", "risk36",
}


def censor_km(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(time)
    time = np.asarray(time, float)[order]
    censored = 1 - np.asarray(event, int)[order]
    unique = np.unique(time)
    survival = []
    value = 1.0
    for point in unique:
        at_risk = np.sum(time >= point)
        censored_at_point = np.sum((time == point) & (censored == 1))
        if at_risk:
            value *= 1.0 - censored_at_point / at_risk
        survival.append(value)
    return unique, np.maximum(np.asarray(survival), 1e-5)


def g_before(km: tuple[np.ndarray, np.ndarray], query: np.ndarray) -> np.ndarray:
    unique, survival = km
    position = np.searchsorted(unique, np.asarray(query), side="left") - 1
    return np.where(position < 0, 1.0, survival[np.maximum(position, 0)])


def ipcw(frame: pd.DataFrame, horizon: int):
    time = frame.time_months.to_numpy(float)
    event = frame.event.to_numpy(int)
    center = frame.center.astype(str).to_numpy()
    cases = (event == 1) & (time <= horizon)
    controls = time > horizon
    weights = np.zeros(len(frame), float)
    for value in np.unique(center):
        in_center = center == value
        km = censor_km(time[in_center], event[in_center])
        center_cases = in_center & cases
        center_controls = in_center & controls
        if center_cases.any():
            weights[center_cases] = 1.0 / g_before(km, time[center_cases])
        weights[center_controls] = 1.0 / float(g_before(km, [horizon])[0])
    return cases, controls, weights


def auc(frame: pd.DataFrame, horizon: int) -> float:
    cases, controls, weights = ipcw(frame, horizon)
    use = cases | controls
    if not cases.any() or not controls.any():
        return float("nan")
    return float(roc_auc_score(
        cases[use].astype(int),
        frame.risk_score.to_numpy(float)[use],
        sample_weight=weights[use],
    ))


def brier(frame: pd.DataFrame, horizon: int) -> float:
    cases, _, weights = ipcw(frame, horizon)
    probability = frame[f"risk{horizon}"].to_numpy(float)
    return float(np.sum(weights * (cases.astype(float) - probability) ** 2) / len(frame))


def calibration(frame: pd.DataFrame, horizon: int = 36) -> dict:
    cases, controls, weights = ipcw(frame, horizon)
    use = cases | controls
    y = cases[use].astype(float)
    probability = np.clip(frame[f"risk{horizon}"].to_numpy(float)[use], 1e-6, 1 - 1e-6)
    local_weights = weights[use]
    logit = np.log(probability / (1 - probability))

    def objective(beta):
        fitted = beta[0] + beta[1] * logit
        return float(np.sum(local_weights * (np.logaddexp(0, fitted) - y * fitted)))

    fit = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    return {
        "intercept": float(fit.x[0]),
        "slope": float(fit.x[1]),
        "converged": bool(fit.success),
    }


def metrics(frame: pd.DataFrame) -> dict:
    return {
        "n": int(len(frame)),
        "events": int(frame.event.sum()),
        "c_index": float(concordance_index(
            frame.time_months, -frame.risk_score, frame.event
        )),
        "auc": {str(h): auc(frame, h) for h in HORIZONS},
        "brier": {str(h): brier(frame, h) for h in HORIZONS},
        "calibration36": calibration(frame),
    }


def draw_center_stratified(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    center = frame.center.astype(str).to_numpy()
    return np.concatenate([
        rng.choice(np.flatnonzero(center == value), size=np.sum(center == value), replace=True)
        for value in np.unique(center)
    ])


def percentile_interval(values: list[float]) -> list[float]:
    values = np.asarray(values, float)
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


def bootstrap(frame: pd.DataFrame, repetitions: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    values = {"auc36": [], "brier36": [], "c_index": []}
    for _ in range(repetitions):
        sample = frame.iloc[draw_center_stratified(frame, rng)].reset_index(drop=True)
        try:
            values["auc36"].append(auc(sample, 36))
            values["brier36"].append(brier(sample, 36))
            values["c_index"].append(float(concordance_index(
                sample.time_months, -sample.risk_score, sample.event
            )))
        except (ValueError, ZeroDivisionError):
            continue
    return {name + "_ci95": percentile_interval(value) for name, value in values.items()}


def validate(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame[list(REQUIRED_COLUMNS - {"patient_id", "center", "model"})].isna().any().any():
        raise ValueError("Required numeric columns contain missing values")


def run(input_path: Path, output_path: Path, repetitions: int, seed: int) -> None:
    predictions = pd.read_csv(input_path)
    validate(predictions)
    result = {
        "method": (
            "Center-stratified external IPCW; censoring survival re-estimated "
            "within each center in every center-stratified patient bootstrap replicate"
        ),
        "bootstrap_repetitions": repetitions,
        "seed": seed,
        "models": {},
    }
    for offset, (model, frame) in enumerate(predictions.groupby("model", sort=True)):
        frame = frame.reset_index(drop=True)
        result["models"][str(model)] = {
            **metrics(frame),
            **bootstrap(frame, repetitions, seed + offset),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    run(args.predictions, args.output, args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
