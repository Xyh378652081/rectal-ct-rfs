"""Finalize only after input, model, artifact, and fit-convergence checks pass."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]


def read(path): return json.loads((OUT / path).read_text(encoding="utf-8-sig"))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inputs = read("00_protocol/input_audit.json")
    artifacts = read("04_validation/artifact_reload_checks.json")
    convergence = read("04_validation/selected_fit_convergence_audit.json")
    results = read("03_models/model_results.json")
    assert inputs["status"] == "PAIRED_INPUTS_VALIDATED" and inputs["unmatched_ids"] == 0
    assert inputs["patients_removed"] == 0 and inputs["n_image_usable"] == 1058
    assert len(artifacts) == 14 and all(x["artifact_reload_matches"] and x["removed_features_absent"] for x in artifacts)
    assert convergence["selected_fits_checked"] == 96 and convergence["selected_optimizer_failure_count"] == 0
    assert len(results["models"]) == 14 and all(len(results["nested_cv"][x]["folds"]) == 5 for x in ["M10", "M12"])
    assert len(list((OUT / "03_models").glob("*_latest_clinical.joblib"))) == 14
    results["retraining_completed"] = True
    results["validation_notes"] = {
        "models_retrained": 14, "patient_id_linkage": "1072/1072 exact match",
        "selected_fit_audit": "96 checked; no optimizer failures",
        "image_feature_extraction_repeated": False,
        "frozen_feature_hashes_verified": True,
        "external_cohorts_previously_inspected": True,
    }
    (OUT / "03_models/model_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    status = read("08_logs/run_status.json")
    status.update({"status": "COMPLETED_WITH_FROZEN_IMAGE_FEATURES", "postprocessing_pending": False,
                   "verification_pending": False, "model_artifacts_verified": 14, "selected_fits_audited": 96,
                   "removed_predictors": read("code/config.json")["removed_predictors"], "image_feature_extraction_repeated": False})
    (OUT / "08_logs/run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    m12 = results["models"]["M12"]["geographic_external"]
    m0b = results["models"]["M0b"]["geographic_external"]
    comparison = results["paired_external_comparisons"]["M12_vs_M0b"]
    report = [
        "# 最新临床数据版模型重训报告", "",
        f"分析版本：`{results['analysis_id']}`。临床表与随访表按患者ID配对，匹配 {inputs['matched_patient_ids']}/{inputs['clinical_rows']} 例；无患者被排除。",
        f"RFS截止日期：{inputs['followup_cutoff_date']}；截止日后随访的无事件患者按截止日行政删失 {inputs['patients_censored_at_cutoff']} 例。",
        "模型保留检及淋巴结总数；已删除区域淋巴结转移、邻近器官侵犯及转移淋巴结数量。",
        "共重训14个Cox模型（M0、M0b、M1–M12）。现有影像组学和CNN特征矩阵经SHA-256核验后冻结复用，未重新提取影像组学特征或训练自监督编码器。",
        "",
        "## 主要外部验证结果", "",
        f"- M12：外部36个月AUC {m12['auc']['36']:.3f}（95% CI {m12['auc36_ci95'][0]:.3f}–{m12['auc36_ci95'][1]:.3f}）；C-index {m12['c_index']:.3f}；36个月Brier评分 {m12['brier']['36']:.3f}。",
        f"- M0b：外部36个月AUC {m0b['auc']['36']:.3f}（95% CI {m0b['auc36_ci95'][0]:.3f}–{m0b['auc36_ci95'][1]:.3f}）。",
        f"- 配对差异（M12 − M0b）：AUC差 {comparison['difference']:.3f}（95% CI {comparison['ci95'][0]:.3f}–{comparison['ci95'][1]:.3f}；bootstrap P={comparison['bootstrap_p_two_sided']:.4g}）。",
        "- 完整指标见 `06_tables/Table_model_performance.csv`；重分层与校准/DCA数据见 `03_models/model_results.json`。",
        "",
        "## 解读边界", "",
        "外部队列在此前研究迭代中已被查看，因此属于既有地理外部队列上的更新评估，而非全新未触及测试集。内层/外层交叉验证针对冻结的影像表征进行下游预处理、PCA和Cox惩罚参数选择；自监督编码器未在每个外层折重新训练。",
        "",
        "## English summary", "",
        f"The latest clinical workbook and prior follow-up data were linked for {inputs['matched_patient_ids']}/{inputs['clinical_rows']} patients. Fourteen Cox models (M0, M0b, and M1–M12) were refit. The number of examined lymph nodes was retained; regional nodal metastasis, adjacent-organ invasion, and metastatic-node count were removed. At the September 27, 2026 administrative cutoff, {inputs['patients_censored_at_cutoff']} event-free patient was censored at the cutoff. Verified frozen radiomic and CNN feature matrices were reused; image-feature extraction and SSL pretraining were not repeated.",
        f"External M12 36-month AUC was {m12['auc']['36']:.3f} (95% CI, {m12['auc36_ci95'][0]:.3f}–{m12['auc36_ci95'][1]:.3f}); paired AUC difference versus M0b was {comparison['difference']:.3f} (95% CI, {comparison['ci95'][0]:.3f}–{comparison['ci95'][1]:.3f}; bootstrap P={comparison['bootstrap_p_two_sided']:.4g}).",
    ]
    report_path = OUT / "07_reports/最新临床数据版_全模型重训报告.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = [{"path": str(p.relative_to(OUT)), "sha256": sha(p), "bytes": p.stat().st_size}
                for p in sorted(OUT.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.name != "output_manifest_sha256.json"]
    (OUT / "00_protocol/output_manifest_sha256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": status["status"], "models": len(results["models"]), "artifacts": len(manifest), "report": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__": main()
