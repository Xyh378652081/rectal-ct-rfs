# Release changes — v1.1.0 (2026-10-06)

- Added the locked external-validation implementation using center-specific censoring distributions and center-stratified bootstrap resampling.
- Updated the model card to the locked external 36-month AUC of 0.917 (95% CI, 0.866–0.954), C-index of 0.819, and Brier score of 0.175.

- Replaced the previous M12 export with the frozen fit from `RECTAL_LATEST_CLINICAL_V3_20261001`.
- Updated the public input schema and deterministic preprocessing to 13 clinical predictors. Examined lymph-node count remains; regional nodal metastasis, adjacent-organ invasion, and metastatic-node count are excluded.
- Updated the fitted design from 58 to 55 features (23 encoded clinical, 20 radiomic, 12 CNN).
- Copied the six cohort-analysis Python sources byte-for-byte from the latest local run. Their configuration is separately sanitized; private workbooks, feature matrices, patient-level outputs, and the original lifelines fit are not shipped.
- Reused the same development-only self-supervised encoder checkpoint as the latest model update; its SHA-256 is reported in `VERIFICATION_REPORT.md`.
- Regenerated the schema and fully synthetic inference input, then verified private/public M12 prediction equivalence locally on 262 external records.

Replace repository and DOI placeholders only after those records exist. Earlier review package `v1.0.0` remains unchanged.
