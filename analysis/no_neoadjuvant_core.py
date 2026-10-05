from __future__ import annotations

import hashlib
import json
import platform
import time
import joblib
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
OUT = Path(CFG["output_root"])
TABLE = OUT / "01_input_qc" / "analysis_table.csv"
RAD = Path(CFG["feature_sources"]["pyradiomics"])
ENG = Path(CFG["feature_sources"]["engineered_spatial"])
CNN = Path(CFG["feature_sources"]["cnn"])
HORIZONS = [12.0, 24.0, 36.0]
SEED = int(CFG["validation"]["random_seeds"][0])
PENALTIES = [float(x) for x in CFG["validation"]["cox_penalizer_grid"]]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def map_binary(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    mapping = {"是": 1.0, "有": 1.0, "阳性": 1.0, "否": 0.0, "无": 0.0, "阴性": 0.0}
    return text.map(mapping).where(text.map(mapping).notna(), pd.to_numeric(values, errors="coerce"))


class ClinicalTransformer:
    continuous = ["age", "tumor_size_cm", "distance_from_anal_verge_cm", "examined_node_count", "cea", "ca19_9"]
    categorical = ["sex", "differentiation", "margin_status", "pni", "lvi", "pt_stage", "pn_stage"]

    def __init__(self):
        continuous_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        categorical_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"))])
        self.pipe = ColumnTransformer([("continuous", continuous_pipe, self.continuous), ("categorical", categorical_pipe, self.categorical)], remainder="drop")

    @staticmethod
    def clean(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for name in ClinicalTransformer.continuous:
            out[name] = pd.to_numeric(out[name], errors="coerce")
        for name in ["cea", "ca19_9"]:
            out[name] = np.log1p(np.clip(out[name], 0, None))
        for name in ["margin_status", "pni", "lvi"]:
            out[name] = map_binary(out[name]).astype("Int64").astype(str)
        for name in ClinicalTransformer.categorical:
            out[name] = out[name].astype(str).str.strip()
        return out

    def fit(self, frame: pd.DataFrame):
        self.pipe.fit(self.clean(frame))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipe.transform(self.clean(frame)), dtype=float)


class ImageTransformer:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.keep: np.ndarray | None = None

    def fit(self, x: np.ndarray):
        x = self.imputer.fit_transform(x)
        self.keep = np.nanstd(x, axis=0) > 1e-8
        x = self.scaler.fit_transform(x[:, self.keep])
        n = min(self.n_components, x.shape[1], max(1, x.shape[0] - 1))
        self.pca = PCA(n_components=n, random_state=SEED).fit(x)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = self.imputer.transform(x)
        x = self.scaler.transform(x[:, self.keep])
        return self.pca.transform(x)


@dataclass
class Sources:
    table: pd.DataFrame
    radiomics: pd.DataFrame
    engineered: pd.DataFrame
    cnn: pd.DataFrame
    ids: np.ndarray
    center: np.ndarray
    time: np.ndarray
    event: np.ndarray
    image_available: np.ndarray
    feature_groups: dict[str, list[str]]


def load_sources() -> Sources:
    table = pd.read_csv(TABLE, encoding="utf-8-sig", low_memory=False)
    forbidden = set(CFG["removed_predictors"]) & set(table.columns)
    if forbidden:
        raise ValueError(f"Removed predictors must not be present: {sorted(forbidden)}")
    table["patient_id"] = table["patient_id"].astype(str).str.strip()
    ids = table["patient_id"].to_numpy(dtype=str)
    rad = pd.read_csv(RAD, encoding="utf-8-sig", low_memory=False).rename(columns={"patient_id": "source_image_patient_id", "research_patient_id": "patient_id"})
    rad["patient_id"] = rad["patient_id"].astype(str).str.strip()
    rad = rad.drop_duplicates("patient_id").set_index("patient_id").reindex(ids)
    eng = pd.read_csv(ENG, encoding="utf-8-sig", low_memory=False)
    eng["patient_id"] = eng["patient_id"].astype(str).str.strip()
    eng = eng.drop_duplicates("patient_id").set_index("patient_id").reindex(ids)
    cnn = pd.read_csv(CNN, encoding="utf-8-sig", low_memory=False)
    cnn["patient_id"] = cnn["patient_id"].astype(str).str.strip()
    cnn = cnn.drop_duplicates("patient_id").set_index("patient_id").reindex(ids)

    rad_numeric = [c for c in rad.columns if c.startswith(("tumor__", "wall__", "nodes__", "spatial_"))]
    tumor = [c for c in rad_numeric if c.startswith("tumor__")]
    wall = [c for c in rad_numeric if c.startswith("wall__")]
    nodes = [c for c in rad_numeric if c.startswith("nodes__")]
    spatial = [c for c in rad_numeric if c.startswith("spatial_")]
    engineered_cross = [c for c in eng.columns if c.startswith(("tumor_wall_", "tumor_nodes_", "wall_nodes_", "peritumor_", "union_"))]
    # Prefix engineered features to prevent collisions when appended to PyRadiomics.
    eng = eng.rename(columns={c: f"eng__{c}" for c in engineered_cross})
    engineered_cross = [f"eng__{c}" for c in engineered_cross]
    feature_groups = {
        "tumor": tumor,
        "tumor_wall": tumor + wall + spatial,
        "tumor_nodes": tumor + nodes + spatial,
        "three": tumor + wall + nodes + spatial,
        "full_rad": tumor + wall + nodes + spatial,
        "engineered_cross": engineered_cross,
        "cnn": [c for c in cnn.columns if c.startswith("cnn_")],
    }
    available = table["image_usable"].astype(str).str.lower().isin(["true", "1"]).to_numpy().copy()
    available &= rad[feature_groups["tumor"]].notna().any(axis=1).to_numpy()
    available &= cnn[feature_groups["cnn"]].notna().any(axis=1).to_numpy()
    return Sources(
        table=table,
        radiomics=rad,
        engineered=eng,
        cnn=cnn,
        ids=ids,
        center=table["center"].to_numpy(dtype=str),
        time=table["rfs_time_months"].to_numpy(dtype=float),
        event=table["rfs_event"].to_numpy(dtype=int),
        image_available=available,
        feature_groups=feature_groups,
    )


MODEL_LABELS = {
    "M0": "临床/病理（全部病例）", "M0b": "临床/病理（影像匹配）", "M1": "肿瘤单结构影像组学",
    "M2": "肿瘤+肠壁", "M3": "肿瘤+淋巴结", "M4": "三结构+空间关系", "M5": "完整多结构影像组学",
    "M6": "CNN深度特征", "M7": "临床+肿瘤", "M8": "临床+肿瘤+肠壁", "M9": "临床+三结构",
    "M10": "临床+完整多结构影像组学", "M11": "临床+CNN", "M12": "临床+影像组学+CNN",
}


def spec_blocks(model: str) -> list[str]:
    return {
        "M0": ["clinical"], "M0b": ["clinical"], "M1": ["tumor"], "M2": ["tumor_wall"],
        "M3": ["tumor_nodes"], "M4": ["three"], "M5": ["full"], "M6": ["cnn"],
        "M7": ["clinical", "tumor"], "M8": ["clinical", "tumor_wall"], "M9": ["clinical", "three"],
        "M10": ["clinical", "full"], "M11": ["clinical", "cnn"], "M12": ["clinical", "full", "cnn"],
    }[model]


def model_mask(data: Sources, model: str) -> np.ndarray:
    return np.ones(len(data.ids), dtype=bool) if model == "M0" else data.image_available.copy()


def raw_image(data: Sources, block: str) -> tuple[np.ndarray, int]:
    pc = CFG["validation"]["radiomics_pca_components"]
    if block == "cnn":
        return data.cnn[data.feature_groups["cnn"]].apply(pd.to_numeric, errors="coerce").to_numpy(float), int(CFG["validation"]["cnn_pca_components"])
    group = "full_rad" if block == "full" else block
    x = data.radiomics[data.feature_groups[group]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if block == "full":
        cross = data.engineered[data.feature_groups["engineered_cross"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        x = np.column_stack([x, cross])
    key = "full" if block == "full" else ("three_structure" if block == "three" else block)
    return x, int(pc[key])


def make_design(data: Sources, train_idx: np.ndarray, test_idx: np.ndarray, model: str):
    train_parts, test_parts, fitted, metadata = [], [], {}, {}
    for block in spec_blocks(model):
        if block == "clinical":
            transformer = ClinicalTransformer().fit(data.table.iloc[train_idx])
            xt, xv = transformer.transform(data.table.iloc[train_idx]), transformer.transform(data.table.iloc[test_idx])
        else:
            raw, n_pc = raw_image(data, block)
            transformer = ImageTransformer(n_pc).fit(raw[train_idx])
            xt, xv = transformer.transform(raw[train_idx]), transformer.transform(raw[test_idx])
            metadata[block] = {"raw_features": int(raw.shape[1]), "pca_components": int(xt.shape[1]), "explained_variance": float(transformer.pca.explained_variance_ratio_.sum())}
        fitted[block] = transformer
        train_parts.append(xt); test_parts.append(xv)
    train = np.nan_to_num(np.column_stack(train_parts), nan=0.0, posinf=0.0, neginf=0.0)
    test = np.nan_to_num(np.column_stack(test_parts), nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.std(train, axis=0) > 1e-8
    metadata["final_design"] = {"before_variance_filter": int(train.shape[1]), "after_variance_filter": int(keep.sum())}
    names = []
    for block in spec_blocks(model):
        if block == "clinical":
            names.extend([f"clinical::{name}" for name in fitted[block].pipe.get_feature_names_out()])
        else:
            names.extend([f"{block}::PC{i+1:02d}" for i in range(fitted[block].pca.n_components_)])
    fitted["_final_keep"] = keep
    metadata["feature_names"] = [name for name, flag in zip(names, keep) if flag]
    assert not any(any(block in name for block in CFG["removed_predictors"]) for name in metadata["feature_names"])
    return train[:, keep], test[:, keep], fitted, metadata


def fit_cox(x: np.ndarray, time: np.ndarray, event: np.ndarray, penalty: float) -> CoxPHFitter:
    frame = pd.DataFrame(x, columns=[f"x{i:03d}" for i in range(x.shape[1])])
    frame["time"] = time; frame["event"] = event
    model = CoxPHFitter(penalizer=penalty)
    model.fit(frame, duration_col="time", event_col="event", show_progress=False)
    return model


def predict(model: CoxPHFitter, x: np.ndarray):
    frame = pd.DataFrame(x, columns=[f"x{i:03d}" for i in range(x.shape[1])])
    score = model.predict_partial_hazard(frame).to_numpy().ravel()
    probability = 1.0 - model.predict_survival_function(frame, times=HORIZONS).to_numpy().T
    return score, probability


def km_censor(time: np.ndarray, event: np.ndarray):
    order = np.argsort(time); t = time[order]; cens = 1 - event[order]
    uniq = np.unique(t); surv = []; s = 1.0
    for value in uniq:
        at = np.sum(t >= value); d = np.sum((t == value) & (cens == 1))
        if at: s *= 1.0 - d / at
        surv.append(s)
    return uniq, np.maximum(np.asarray(surv), 1e-5)


def g_before(km, query):
    uniq, surv = km; pos = np.searchsorted(uniq, np.asarray(query), side="left") - 1
    return np.where(pos < 0, 1.0, surv[np.maximum(pos, 0)])


def td_auc(time, event, score, horizon, censor_km):
    cases = (event == 1) & (time <= horizon); controls = time > horizon
    if not cases.any() or not controls.any(): return np.nan
    case_scores, control_scores = score[cases], score[controls]
    weights = 1.0 / g_before(censor_km, time[cases])
    wins = (case_scores[:, None] > control_scores[None, :]).sum(axis=1) + 0.5 * (case_scores[:, None] == control_scores[None, :]).sum(axis=1)
    return float(np.sum(weights * wins) / (np.sum(weights) * len(control_scores)))


def brier(time, event, probability, horizon, censor_km):
    cases = (event == 1) & (time <= horizon); controls = time > horizon
    weights = np.zeros(len(time)); y = np.zeros(len(time)); y[cases] = 1
    weights[cases] = 1.0 / g_before(censor_km, time[cases])
    weights[controls] = 1.0 / float(g_before(censor_km, [horizon])[0])
    return float(np.sum(weights * (y - probability) ** 2) / len(time))


def calibration(time, event, probability, horizon, censor_km):
    cases = (event == 1) & (time <= horizon); controls = time > horizon; use = cases | controls
    y = cases[use].astype(float); p = np.clip(probability[use], 1e-6, 1 - 1e-6)
    w = np.where(cases[use], 1 / g_before(censor_km, time[use]), 1 / g_before(censor_km, [horizon])[0])
    logit = np.log(p / (1 - p))
    def objective(beta):
        z = beta[0] + beta[1] * logit
        return float(np.sum(w * (np.logaddexp(0, z) - y * z)))
    fit = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    groups = np.array_split(np.argsort(p), min(10, len(p)))
    curve = [{"n": int(len(g)), "predicted": float(np.average(p[g], weights=w[g])), "observed": float(np.average(y[g], weights=w[g]))} for g in groups]
    return {"intercept": float(fit.x[0]), "slope": float(fit.x[1]), "converged": bool(fit.success), "curve": curve}


def dca(time, event, probability, horizon, censor_km):
    cases = (event == 1) & (time <= horizon); controls = time > horizon; use = cases | controls
    y = cases[use].astype(float); p = probability[use]
    w = np.where(cases[use], 1 / g_before(censor_km, time[use]), 1 / g_before(censor_km, [horizon])[0])
    rows = []
    for threshold in np.arange(0.05, 0.51, 0.01):
        treat = p >= threshold
        nb = (np.sum(w * treat * y) - np.sum(w * treat * (1 - y)) * threshold / (1 - threshold)) / len(time)
        all_nb = (np.sum(w * y) - np.sum(w * (1 - y)) * threshold / (1 - threshold)) / len(time)
        rows.append({"threshold": float(threshold), "model": float(nb), "treat_all": float(all_nb), "treat_none": 0.0})
    return rows


def evaluate(time, event, score, probability, censor_reference_time, censor_reference_event, bootstrap=0, seed=SEED):
    km = km_censor(censor_reference_time, censor_reference_event)
    result = {"n": int(len(time)), "events": int(event.sum()), "c_index": float(concordance_index(time, -score, event)), "auc": {}, "brier": {}}
    for j, horizon in enumerate(HORIZONS):
        result["auc"][str(int(horizon))] = td_auc(time, event, score, horizon, km)
        result["brier"][str(int(horizon))] = brier(time, event, probability[:, j], horizon, km)
    result["calibration36"] = calibration(time, event, probability[:, 2], 36.0, km)
    result["dca36"] = dca(time, event, probability[:, 2], 36.0, km)
    if bootstrap:
        rng = np.random.default_rng(seed); aucs, cidx = [], []
        for _ in range(bootstrap):
            idx = rng.integers(0, len(time), len(time))
            if 0 < event[idx].sum() < len(idx):
                aucs.append(td_auc(time[idx], event[idx], score[idx], 36.0, km))
                try: cidx.append(concordance_index(time[idx], -score[idx], event[idx]))
                except ZeroDivisionError: pass
        result["auc36_ci95"] = [float(np.nanpercentile(aucs, 2.5)), float(np.nanpercentile(aucs, 97.5))]
        result["c_index_ci95"] = [float(np.nanpercentile(cidx, 2.5)), float(np.nanpercentile(cidx, 97.5))]
    return result


def tune_penalty(data: Sources, indices: np.ndarray, model: str, folds=3, seed=SEED):
    y = data.event[indices]; skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    # Reuse a design only within its own training/validation fold, across penalties.
    designs = []
    for train_pos, valid_pos in skf.split(indices, y):
        train_idx, valid_idx = indices[train_pos], indices[valid_pos]
        xt, xv, _, _ = make_design(data, train_idx, valid_idx, model)
        designs.append((train_idx, valid_idx, xt, xv))
    rows = []
    for penalty in PENALTIES:
        scores = []
        for train_idx, valid_idx, xt, xv in designs:
            fit = fit_cox(xt, data.time[train_idx], data.event[train_idx], penalty)
            score, _ = predict(fit, xv)
            scores.append(float(concordance_index(data.time[valid_idx], -score, data.event[valid_idx])))
        rows.append({"penalizer": penalty, "mean_cindex": float(np.mean(scores)), "fold_cindex": scores})
    best = max(rows, key=lambda x: x["mean_cindex"])
    return float(best["penalizer"]), rows


def fit_final(data: Sources, train_idx: np.ndarray, test_idx: np.ndarray, model_name: str, penalty=None):
    if penalty is None:
        penalty, tuning = tune_penalty(data, train_idx, model_name)
    else: tuning = None
    xt, xv, transformers, metadata = make_design(data, train_idx, test_idx, model_name)
    fit = fit_cox(xt, data.time[train_idx], data.event[train_idx], penalty)
    train_score, train_prob = predict(fit, xt); test_score, test_prob = predict(fit, xv)
    return {"penalizer": penalty, "tuning": tuning, "metadata": metadata, "n_features": int(xt.shape[1]), "train_score": train_score, "train_prob": train_prob, "test_score": test_score, "test_prob": test_prob, "model_object": fit, "transformers": transformers, "train_design": xt, "test_design": xv}


def nested_cv(data: Sources, model_name: str):
    use = model_mask(data, model_name) & (data.center == CFG["development_center"]); idx = np.where(use)[0]
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED); rows = []
    for fold, (train_pos, test_pos) in enumerate(outer.split(idx, data.event[idx]), 1):
        print(json.dumps({"phase": "nested_cv", "model": model_name, "outer_fold": fold}), flush=True)
        train_idx, test_idx = idx[train_pos], idx[test_pos]
        fit = fit_final(data, train_idx, test_idx, model_name)
        metric = evaluate(data.time[test_idx], data.event[test_idx], fit["test_score"], fit["test_prob"], data.time[train_idx], data.event[train_idx])
        rows.append({"fold": fold, "n": len(test_idx), "events": int(data.event[test_idx].sum()), "penalizer": fit["penalizer"], "auc36": metric["auc"]["36"], "c_index": metric["c_index"]})
    return {"model": model_name, "folds": rows, "mean_auc36": float(np.mean([x["auc36"] for x in rows])), "sd_auc36": float(np.std([x["auc36"] for x in rows], ddof=1)), "mean_c_index": float(np.mean([x["c_index"] for x in rows]))}


def paired_difference(time, event, ref, candidate, censor_km, reps=2000, seed=SEED):
    observed = td_auc(time, event, candidate, 36, censor_km) - td_auc(time, event, ref, 36, censor_km)
    rng = np.random.default_rng(seed); values = []
    for _ in range(reps):
        idx = rng.integers(0, len(time), len(time))
        values.append(td_auc(time[idx], event[idx], candidate[idx], 36, censor_km) - td_auc(time[idx], event[idx], ref[idx], 36, censor_km))
    values = np.asarray(values)
    return {"difference": float(observed), "ci95": [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))], "bootstrap_p_two_sided": float(2 * min(np.mean(values <= 0), np.mean(values >= 0)))}


def main():
    data = load_sources(); dev = data.center == CFG["development_center"]; ext = np.isin(data.center, CFG["external_centers"])
    results = {"analysis_id": CFG["analysis_id"], "status": CFG["status"], "endpoint": CFG["endpoint"], "feature_sources": {"pyradiomics": str(RAD), "engineered_spatial": str(ENG), "cnn": str(CNN)}, "feature_hashes": {"pyradiomics": sha256(RAD), "engineered_spatial": sha256(ENG), "cnn": sha256(CNN)}, "removed_predictors": CFG["removed_predictors"], "models": {}, "paired_external_comparisons": {}, "nested_cv": {}}
    prediction_rows = []; stores = {}
    model_sequence = ["M0", "M0b", *[f"M{i}" for i in range(1, 13)]]
    for number, name in enumerate(model_sequence):
        tick = time.monotonic()
        print(json.dumps({"phase": "fit_start", "model": name}), flush=True)
        use = model_mask(data, name); train_idx = np.where(dev & use)[0]; test_idx = np.where(ext & use)[0]
        fitted = fit_final(data, train_idx, test_idx, name)
        development = evaluate(data.time[train_idx], data.event[train_idx], fitted["train_score"], fitted["train_prob"], data.time[train_idx], data.event[train_idx])
        external = evaluate(data.time[test_idx], data.event[test_idx], fitted["test_score"], fitted["test_prob"], data.time[train_idx], data.event[train_idx], bootstrap=2000, seed=SEED + number)
        center_metrics = {}
        for center in CFG["external_centers"]:
            local = data.center[test_idx] == center
            center_metrics[center] = evaluate(data.time[test_idx][local], data.event[test_idx][local], fitted["test_score"][local], fitted["test_prob"][local], data.time[train_idx], data.event[train_idx], bootstrap=500, seed=SEED + number + len(center))
        results["models"][name] = {"label": MODEL_LABELS[name], "n_features_after_preprocessing": fitted["n_features"], "penalizer": fitted["penalizer"], "inner_tuning": fitted["tuning"], "block_metadata": fitted["metadata"], "development_apparent": development, "geographic_external": external, "external_by_center": center_metrics}
        stores[name] = {"train_idx": train_idx, "test_idx": test_idx, **fitted}
        joblib.dump({"analysis_id": CFG["analysis_id"], "model_name": name, "cox_model": fitted["model_object"], "transformers": fitted["transformers"], "feature_names": fitted["metadata"]["feature_names"], "feature_groups": data.feature_groups, "removed_predictors": CFG["removed_predictors"], "clinical_predictors": CFG["clinical_predictors"]}, OUT / "03_models" / f"{name}_latest_clinical.joblib", compress=3)
        (OUT / "03_models" / f"{name}_results.json").write_text(json.dumps(results["models"][name], ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"phase": "fit_done", "model": name, "n_features": fitted["n_features"], "penalty": fitted["penalizer"], "external_auc36": external["auc"]["36"], "elapsed_seconds": round(time.monotonic()-tick, 2)}), flush=True)
        for pos, patient_index in enumerate(test_idx):
            prediction_rows.append({"patient_id": data.ids[patient_index], "center": data.center[patient_index], "model": name, "time_months": data.time[patient_index], "event": data.event[patient_index], "risk_score": fitted["test_score"][pos], "risk12": fitted["test_prob"][pos, 0], "risk24": fitted["test_prob"][pos, 1], "risk36": fitted["test_prob"][pos, 2]})
    for candidate, reference in [("M4", "M1"), ("M10", "M0b"), ("M12", "M0b"), ("M12", "M10")]:
        a, b = stores[candidate], stores[reference]
        if not np.array_equal(a["test_idx"], b["test_idx"]): raise ValueError("Paired cohort mismatch")
        train_idx = a["train_idx"]; km = km_censor(data.time[train_idx], data.event[train_idx]); test_idx = a["test_idx"]
        results["paired_external_comparisons"][f"{candidate}_vs_{reference}"] = paired_difference(data.time[test_idx], data.event[test_idx], b["test_score"], a["test_score"], km, seed=SEED + len(results["paired_external_comparisons"]))
    print(json.dumps({"phase": "paired_comparisons_complete"}), flush=True)
    results["nested_cv"]["M10"] = nested_cv(data, "M10")
    results["nested_cv"]["M12"] = nested_cv(data, "M12")
    pd.DataFrame(prediction_rows).to_csv(OUT / "03_models" / "external_patient_predictions.csv", index=False, encoding="utf-8-sig")
    summary = []
    for name, item in results["models"].items():
        extm = item["geographic_external"]
        summary.append({"model": name, "label": item["label"], "n_features": item["n_features_after_preprocessing"], "penalizer": item["penalizer"], "external_n": extm["n"], "external_events": extm["events"], "external_auc12": extm["auc"]["12"], "external_auc24": extm["auc"]["24"], "external_auc36": extm["auc"]["36"], "auc36_ci_low": extm["auc36_ci95"][0], "auc36_ci_high": extm["auc36_ci95"][1], "external_cindex": extm["c_index"], "external_brier36": extm["brier"]["36"], "calibration_intercept36": extm["calibration36"]["intercept"], "calibration_slope36": extm["calibration36"]["slope"]})
    pd.DataFrame(summary).to_csv(OUT / "06_tables" / "Table_model_performance.csv", index=False, encoding="utf-8-sig")
    results["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    (OUT / "03_models" / "model_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))
    return data, results, stores


if __name__ == "__main__":
    main()
