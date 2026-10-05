"""Reload-check current-version models and derive risk and contribution tables."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test
from threadpoolctl import threadpool_limits

import no_neoadjuvant_core as core

OUT = core.OUT


def safe(value):
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, np.ndarray): return safe(value.tolist())
    if isinstance(value, np.generic): return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def save_json(path, value):
    path.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2), encoding="utf-8")


def holm(values):
    p = np.asarray(values, float); order = np.argsort(p); out = np.empty(len(p)); previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, p[index] * (len(p) - rank))); out[index] = previous
    return out


def transform_bundle(bundle, data, indices):
    parts = []
    for block in core.spec_blocks(bundle["model_name"]):
        transformer = bundle["transformers"][block]
        if block == "clinical": x = transformer.transform(data.table.iloc[indices])
        else: x = transformer.transform(core.raw_image(data, block)[0][indices])
        parts.append(x)
    x = np.nan_to_num(np.column_stack(parts), nan=0.0, posinf=0.0, neginf=0.0)
    return x[:, bundle["transformers"]["_final_keep"]]


def km_summary(frame):
    km = KaplanMeierFitter().fit(frame.time_months, frame.event)
    result = {"n": int(len(frame)), "events": int(frame.event.sum())}
    for horizon in [12, 24, 36]:
        ci = km.confidence_interval_survival_function_.loc[lambda x: x.index <= horizon].iloc[-1].to_numpy(float)
        result[f"rfs{horizon}"] = float(km.predict(horizon))
        result[f"rfs{horizon}_ci_low"] = float(ci[0]); result[f"rfs{horizon}_ci_high"] = float(ci[1])
    return result


def main():
    with threadpool_limits(limits=1):
        data = core.load_sources()
        results = json.loads((OUT / "03_models/model_results.json").read_text(encoding="utf-8"))
        pred = pd.read_csv(OUT / "03_models/external_patient_predictions.csv", dtype={"patient_id": str})
        devpred = pd.read_csv(OUT / "03_models/development_patient_predictions.csv", dtype={"patient_id": str})
        verified = []
        for name, result in results["models"].items():
            use = core.model_mask(data, name)
            train_idx = np.where(use & (data.center == core.CFG["development_center"]))[0]
            test_idx = np.where(use & np.isin(data.center, core.CFG["external_centers"]))[0]
            assert set(data.ids[train_idx]).isdisjoint(set(data.ids[test_idx]))
            bundle = joblib.load(OUT / "03_models" / f"{name}_latest_clinical.joblib")
            excluded = set(core.CFG["removed_predictors"])
            assert not excluded.intersection(bundle["clinical_predictors"] and [x["locked_name"] for x in bundle["clinical_predictors"]])
            assert not any(any(item in feature for item in excluded) for feature in bundle["feature_names"])
            x = transform_bundle(bundle, data, test_idx)
            score, probability = core.predict(bundle["cox_model"], x)
            current = pred[pred.model.eq(name)].set_index("patient_id").loc[data.ids[test_idx]]
            np.testing.assert_allclose(score, current.risk_score, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(probability, current[["risk12", "risk24", "risk36"]], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(current.time_months, data.time[test_idx], rtol=0, atol=1e-10)
            np.testing.assert_array_equal(current.event.to_numpy(dtype=int), data.event[test_idx])
            verified.append({"model": name, "artifact_reload_matches": True, "removed_features_absent": True,
                             "n_features": int(x.shape[1]), "external_n": int(len(test_idx))})
        save_json(OUT / "04_validation/artifact_reload_checks.json", verified)

        secondary = ["M4_vs_M1", "M10_vs_M0b", "M12_vs_M10"]
        comparison_rows = []
        pvals = [results["paired_external_comparisons"][key]["bootstrap_p_two_sided"] for key in secondary]
        adjusted = holm(pvals)
        for key, item in results["paired_external_comparisons"].items():
            row = {"comparison": key, "primary": key == "M12_vs_M0b", **item}
            if key in secondary: row["p_holm_secondary_family"] = float(adjusted[secondary.index(key)])
            comparison_rows.append(row)
        save_json(OUT / "04_validation/within_version_comparisons.json", comparison_rows)

        thresholds = {}
        for name in ["M0b", "M12"]:
            scores = devpred.loc[devpred.model.eq(name), "risk_score"].to_numpy()
            q1, median, q3 = np.quantile(scores, [1/3, 1/2, 2/3])
            thresholds[name] = {"lower_tertile": float(q1), "median": float(median), "upper_tertile": float(q3)}
        m12 = pred[pred.model.eq("M12")].copy().set_index("patient_id")
        m0b = pred[pred.model.eq("M0b")].set_index("patient_id").loc[m12.index]
        base = m12[["center", "time_months", "event"]].copy()
        for name, frame in [("M0b", m0b), ("M12", m12)]:
            lo, median, hi = [thresholds[name][k] for k in ["lower_tertile", "median", "upper_tertile"]]
            prefix = name.lower(); score = frame.risk_score.to_numpy()
            base[f"{prefix}_score"] = score; base[f"{prefix}_risk36"] = frame.risk36.to_numpy()
            base[f"{prefix}_binary"] = np.where(score < median, "Low", "High")
            base[f"{prefix}_tier"] = np.select([score < lo, score < hi], ["Low", "Intermediate"], default="High")
        base["cross_group"] = base.m0b_binary + "/" + base.m12_binary
        base.reset_index().to_csv(OUT / "03_models/external_M0b_M12_risk_groups.csv", index=False, encoding="utf-8-sig")
        group_rows = []
        for grouping in ["m12_binary", "m12_tier", "cross_group"]:
            p = float(multivariate_logrank_test(base.time_months, base[grouping], base.event).p_value)
            for group, frame in base.groupby(grouping, sort=False):
                group_rows.append({"grouping": grouping, "group": group, "global_logrank_p": p, **km_summary(frame)})
        pd.DataFrame(group_rows).to_csv(OUT / "06_tables/Table_updated_risk_stratification.csv", index=False, encoding="utf-8-sig")
        tier_tests = []
        for group_a, group_b in [("Low", "Intermediate"), ("Low", "High"), ("Intermediate", "High")]:
            a = base[base.m12_tier.eq(group_a)]; b = base[base.m12_tier.eq(group_b)]
            tier_tests.append({"group_a": group_a, "group_b": group_b,
                               "p_nominal": float(logrank_test(a.time_months, b.time_months, a.event, b.event).p_value)})
        for row, p in zip(tier_tests, holm([x["p_nominal"] for x in tier_tests])): row["p_holm"] = float(p)
        pd.DataFrame(tier_tests).to_csv(OUT / "06_tables/Table_three_tier_pairwise.csv", index=False, encoding="utf-8-sig")
        save_json(OUT / "04_validation/development_risk_thresholds.json", thresholds)

        bundle = joblib.load(OUT / "03_models/M12_latest_clinical.joblib")
        matrices = np.load(OUT / "03_models/M12_design_matrices.npz", allow_pickle=False)
        train_matrix, external_matrix = matrices["train"], matrices["external"]
        train_idx, external_idx = matrices["train_indices"], matrices["external_indices"]
        beta = bundle["cox_model"].params_.to_numpy(float); names = bundle["feature_names"]
        train_contrib, external_contrib = train_matrix * beta, external_matrix * beta
        feature_table = pd.DataFrame({"feature": names, "block": [x.split("::")[0] for x in names],
            "coefficient": beta, "development_mean_abs_contribution": np.abs(train_contrib).mean(axis=0),
            "external_mean_abs_contribution": np.abs(external_contrib).mean(axis=0)}).sort_values("development_mean_abs_contribution", ascending=False)
        feature_table["development_rank"] = np.arange(1, len(feature_table) + 1)
        feature_table.to_csv(OUT / "06_tables/Table_M12_feature_contributions.csv", index=False, encoding="utf-8-sig")
        block_table = pd.DataFrame({"patient_id": data.ids[external_idx], "center": data.center[external_idx],
                                    "time_months": data.time[external_idx], "event": data.event[external_idx]})
        for block in ["clinical", "full", "cnn"]:
            mask = np.array([x.startswith(block + "::") for x in names])
            block_table[f"{block}_signed_contribution"] = external_contrib[:, mask].sum(axis=1)
            block_table[f"{block}_sum_abs_feature_contribution"] = np.abs(external_contrib[:, mask]).sum(axis=1)
        block_table["uncentered_linear_predictor"] = external_contrib.sum(axis=1)
        np.testing.assert_allclose(block_table[[f"{b}_signed_contribution" for b in ["clinical", "full", "cnn"]]].sum(axis=1), external_contrib.sum(axis=1), atol=1e-10)
        block_table.to_csv(OUT / "03_models/external_M12_block_contributions.csv", index=False, encoding="utf-8-sig")
        full = pd.DataFrame(external_contrib, columns=names); full.insert(0, "patient_id", data.ids[external_idx])
        full.to_csv(OUT / "03_models/external_M12_all_feature_contributions.csv", index=False, encoding="utf-8-sig")

        report = {"models_verified": verified, "primary_comparison": results["paired_external_comparisons"]["M12_vs_M0b"],
                  "thresholds": thresholds, "risk_groups": group_rows, "nested_cv": results["nested_cv"],
                  "input_and_artifact_checks_passed": True,
                  "censoring_cutoff": core.CFG["followup_cutoff_date"],
                  "removed_predictors": core.CFG["removed_predictors"],
                  "calibration_convergence_flags": {name: item["geographic_external"]["calibration36"]["converged"] for name, item in results["models"].items()}}
        save_json(OUT / "07_reports/retraining_summary.json", report)
        status = json.loads((OUT / "08_logs/run_status.json").read_text(encoding="utf-8"))
        status["postprocessing_pending"] = False; status["verification_pending"] = "selected-fit convergence audit"
        save_json(OUT / "08_logs/run_status.json", status)
        print(json.dumps(safe({"M12_auc36": results["models"]["M12"]["geographic_external"]["auc"]["36"],
                               "primary_comparison": report["primary_comparison"], "models_verified": len(verified),
                               "tier_tests": tier_tests}), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
