# Latest clinical-data model update: cohort-level code

The model-development sources for analysis `RECTAL_LATEST_CLINICAL_V3_20261001` are included here: `prepare_inputs.py`, `no_neoadjuvant_core.py`, `train_models.py`, `summarize_update.py`, `audit_selected_fits.py`, and `finalize_run.py`. The final manuscript metrics were subsequently locked with center-specific external censoring distributions and center-stratified bootstrap resampling; that final evaluation is implemented in `locked_external_validation.py`.

The scripts need institution-approved clinical/follow-up tables, the prior linkage snapshot, and frozen patient-level radiomic/CNN feature matrices. None of those patient-level inputs or outputs is included in this public package. `config.json` is a sanitized reproduction template with paths under `private/`; it records the original input SHA-256 values but does not contain the source files. Place authorized inputs outside version control and adjust paths as needed. The exact published cohort cannot be reconstructed from this package alone.

Run the staged pipeline only in an approved secure environment:

```bash
python analysis/prepare_inputs.py
python analysis/train_models.py
python analysis/summarize_update.py
python analysis/audit_selected_fits.py
python analysis/finalize_run.py
```

To reproduce the locked external metric calculation from an authorized patient-level prediction table:

```bash
python analysis/locked_external_validation.py \
  --predictions private/external_patient_predictions.csv \
  --output private/locked_external_metrics.json \
  --bootstrap 1200 \
  --seed 20261004
```

The required columns are `patient_id`, `center`, `model`, `time_months`, `event`, `risk_score`, `risk12`, `risk24`, and `risk36`. The prediction table and resulting patient-level materials are excluded from this repository.

This release does **not** claim complete nested validation of self-supervised encoder training. The encoder and image features were frozen and reused in the latest clinical-data model update. The packaged M12 inference example is independent of private patient data; use `scripts/predict.py` with `examples/synthetic_input.csv` for a smoke test.
