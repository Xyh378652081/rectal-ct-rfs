# Verification report — latest clinical-data model

- Analysis identifier: `RECTAL_LATEST_CLINICAL_V3_20261001`.
- Source artifact: the locally fitted `M12_latest_clinical.joblib`; this private artifact is **not** distributed.
- Public M12 design: 13 clinical inputs, 392 radiomic-spatial inputs, and 128 frozen CNN inputs; 55 fitted features after preprocessing.
- Public model export: completed; original lifelines training-frame fields and patient-level follow-up/event data are omitted.
- External equivalence check performed locally on 262 approved records without saving patient-level output to this package.
- Maximum absolute difference in the 262 × 55 transformed design: `2.842170943040401e-14`.
- Maximum absolute difference in event risk versus the private run: 12 months `1.6653345369377348e-16`; 24 months `4.440892098500626e-16`; 36 months `2.220446049250313e-16`.
- Frozen self-supervised encoder SHA-256 matches the development-only source checkpoint: `e69615ce7f307ae58ddbdf19457ebd89c3d27ce02bb38718db1dfdd8919c0819`.
- Synthetic inference smoke test: passed.

The equivalence calculations used private local data only. Neither the private input records nor their predictions or design matrices are included in this release. The release manifest records package-file hashes; the source clinical/follow-up files and frozen patient-level imaging-feature matrices are only identified by hashes in the sanitized configuration.
