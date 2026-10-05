"""Pair latest clinical data with verified prior follow-up and frozen image features."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
OUT = Path(CFG["output_root"])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_book(path, id_col):
    frame = pd.read_excel(path, dtype={id_col: str})
    frame[id_col] = frame[id_col].astype(str).str.strip()
    if frame[id_col].isna().any() or frame[id_col].duplicated().any():
        raise ValueError(f"Missing or duplicate patient IDs in {Path(path).name}")
    return frame


def normalize_center(value):
    text = str(value).strip().upper()
    if text.startswith("CENTER1"):
        return "Center1"
    if text.startswith("CENTER2"):
        return "Center2"
    if text.startswith("CENTER3"):
        return "Center3"
    if text.startswith("CENTER4"):
        return "Center4"
    raise ValueError(f"Unrecognized center code: {value!r}")


def main():
    for sub in ["00_protocol", "01_input_qc", "03_models", "04_validation", "05_figures", "06_tables", "07_reports", "08_logs"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    clinical_path = Path(CFG["clinical_source"])
    followup_path = Path(CFG["followup_source"])
    clinical_hash = digest(clinical_path)
    followup_hash = digest(followup_path)
    assert clinical_hash == CFG["clinical_source_expected_sha256"], "Latest clinical workbook changed after audit."
    assert followup_hash == CFG["followup_source_expected_sha256"], "Follow-up workbook changed after audit."

    id_col = CFG["source_id_name"]
    clinical = read_book(clinical_path, id_col).set_index(id_col, drop=False)
    followup = read_book(followup_path, id_col).set_index(id_col, drop=False)
    prev_path = Path(CFG["prior_snapshot_source"])
    assert digest(prev_path) == CFG["prior_snapshot_expected_sha256"], "Prior snapshot changed after linkage audit."
    prior = pd.read_csv(prev_path, dtype={"patient_id": str}, encoding="utf-8-sig").set_index("patient_id")

    clinical_ids, followup_ids, prior_ids = set(clinical.index), set(followup.index), set(prior.index)
    assert clinical_ids == followup_ids == prior_ids, "Patient-level source linkage is not one-to-one."
    ids = clinical.index.astype(str)
    followup = followup.loc[ids]
    prior = prior.loc[ids]

    center = clinical[CFG["source_center_name"]].map(normalize_center)
    assert np.array_equal(center.to_numpy(dtype=str), prior["center"].to_numpy(dtype=str)), "Center assignments differ across sources."

    event = pd.to_numeric(followup[CFG["endpoint"]["source_event_name"]], errors="raise").astype(int)
    time_days = pd.to_numeric(followup[CFG["endpoint"]["source_time_name"]], errors="raise").astype(float)
    np.testing.assert_array_equal(event.to_numpy(), prior["rfs_event"].to_numpy(dtype=int))
    np.testing.assert_allclose(time_days.to_numpy(), prior["rfs_time_days"].to_numpy(dtype=float), rtol=0, atol=1e-9)

    cutoff = pd.Timestamp(CFG["followup_cutoff_date"])
    surgery_date = pd.to_datetime(clinical["手术日期"], errors="raise")
    last_followup = pd.to_datetime(followup["末次随访日期"], errors="raise")
    if surgery_date.isna().any() or last_followup.isna().any():
        raise ValueError("Surgery date or last-follow-up date is missing.")
    max_followup_days = (cutoff - surgery_date).dt.total_seconds() / 86400.0
    if ((event == 1) & (time_days > max_followup_days + 1e-9)).any():
        raise ValueError("An RFS event occurs after the prespecified follow-up cutoff.")
    censor_after_cutoff = (event == 0) & (last_followup > cutoff)
    time_days.loc[censor_after_cutoff] = max_followup_days.loc[censor_after_cutoff]
    if (time_days <= 0).any():
        raise ValueError("Nonpositive follow-up time after applying the cutoff.")

    table = pd.DataFrame({
        "patient_id": ids,
        "center": center.to_numpy(dtype=str),
        "rfs_event": event.to_numpy(dtype=int),
        "rfs_time_days": time_days.to_numpy(dtype=float),
        "image_usable": prior["image_usable"].to_numpy(),
        "ct_date": pd.to_datetime(clinical[CFG["source_ct_date_name"]], errors="raise").dt.strftime("%Y-%m-%d").to_numpy(),
    })
    table["rfs_time_months"] = table["rfs_time_days"] / CFG["endpoint"]["days_per_month"]
    table["analysis_role"] = np.where(table.center.eq(CFG["development_center"]), "development", "geographic_external")

    for item in CFG["clinical_predictors"]:
        locked, source = item["locked_name"], item["source_name"]
        source_frame = followup if locked == "examined_node_count" else clinical
        if source not in source_frame.columns:
            raise ValueError(f"Required retained predictor absent: {source}")
        table[locked] = source_frame[source].to_numpy()

    removed = set(CFG["removed_predictors"]) & set(table.columns)
    if removed:
        raise ValueError(f"Removed predictors unexpectedly present: {sorted(removed)}")
    if "0" in table["pt_stage"].astype(str).str.strip().unique():
        raise ValueError("Unexpected pT0 remains in the September 12 clinical data.")
    assert len(table) == 1072 and table.patient_id.is_unique
    assert table.rfs_event.isin([0, 1]).all()
    assert table.image_usable.astype(str).str.lower().isin(["true", "1"]).sum() == 1058
    assert int(censor_after_cutoff.sum()) == 1

    feature_hashes = {key: digest(path) for key, path in CFG["feature_sources"].items()}
    assert feature_hashes == CFG["expected_feature_hashes"], "Frozen imaging feature files changed."
    table.to_csv(OUT / "01_input_qc" / "analysis_table.csv", index=False, encoding="utf-8-sig")
    counts = table.groupby(["center", "analysis_role"]).agg(n=("patient_id", "size"), events=("rfs_event", "sum"), image_usable=("image_usable", "sum")).reset_index()
    counts.to_csv(OUT / "06_tables" / "cohort_counts.csv", index=False, encoding="utf-8-sig")
    report = {
        "analysis_id": CFG["analysis_id"], "status": "PAIRED_INPUTS_VALIDATED",
        "clinical_source": str(clinical_path), "clinical_source_sha256": clinical_hash,
        "followup_source": str(followup_path), "followup_source_sha256": followup_hash,
        "prior_snapshot_sha256": digest(prev_path), "analysis_table_sha256": digest(OUT / "01_input_qc" / "analysis_table.csv"),
        "clinical_rows": len(clinical), "followup_rows": len(followup), "prior_snapshot_rows": len(prior),
        "matched_patient_ids": len(clinical_ids), "unmatched_ids": 0,
        "rfs_event_count": int(event.sum()), "patients_censored_at_cutoff": int(censor_after_cutoff.sum()),
        "followup_cutoff_date": CFG["followup_cutoff_date"], "n_image_usable": int(table.image_usable.astype(str).str.lower().isin(["true", "1"]).sum()),
        "retained_clinical_predictors": [x["locked_name"] for x in CFG["clinical_predictors"]],
        "removed_predictors": CFG["removed_predictors"], "patients_removed": 0,
        "frozen_image_feature_hashes": feature_hashes,
        "limitations": ["CNN encoder and radiomic feature extraction were not repeated; verified frozen patient-level feature matrices were reused.",
                        "External cohorts have been inspected in prior analyses and are not new unseen test cohorts.",
                        "Internal nested validation covers downstream imputation, scaling, PCA, and Cox tuning conditional on the frozen image representations."]
    }
    (OUT / "00_protocol" / "input_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["status", "clinical_rows", "followup_rows", "matched_patient_ids", "unmatched_ids", "rfs_event_count", "patients_censored_at_cutoff", "n_image_usable", "analysis_table_sha256"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
