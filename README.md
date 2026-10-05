# Rectal CT Recurrence-Risk Model

This review package accompanies the locked four-center rectal cancer recurrence-risk study (model fit `RECTAL_LATEST_CLINICAL_V3_20261001`; final external evaluation locked October 4, 2026). The model combines preoperative CT-derived representations with clinicopathologic information available after surgery; it is not a preoperative-only decision model.

It provides a privacy-preserving research implementation of the frozen full-fusion model (M12), including preprocessing parameters, Cox model parameters, the frozen self-supervised 3D CNN encoder, core radiomic-spatial feature definitions, input schemas, and an inference command. The public model files do **not** contain patient identifiers, individual follow-up times, outcomes, CT images, segmentation masks, or patient-level predictions.

## Scope

- Endpoint: recurrence-free survival after curative-intent surgery.
- Event: first local recurrence, distant metastasis, or death from any cause.
- Outputs: relative risk and absolute event risk at 12, 24, and 36 months.
- Inputs: 13 clinicopathologic variables, 392 multistructure radiomic-spatial features, and 128 frozen CNN embedding features. The fitted M12 design has 23 encoded clinical features, 20 radiomic principal components, and 12 CNN principal components (55 total).
- Status: research use only; not a medical device and not intended for direct clinical decision-making.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

For CT-to-feature extraction and the frozen 3D CNN encoder, additionally install `requirements-imaging.txt`.

## Quick start

Generate a fully synthetic input row and run the frozen model:

```bash
python examples/make_synthetic_input.py
python scripts/predict.py --input examples/synthetic_input.csv --output examples/synthetic_prediction.csv
```

The synthetic example is provided only to test execution. It is not a clinical case and its output must not be interpreted clinically.

The final manuscript's external metrics used center-specific censoring distributions and a center-stratified patient bootstrap. The privacy-preserving implementation is `analysis/locked_external_validation.py`. It accepts patient-level predictions only inside an institution-approved secure environment; no patient-level prediction table is included in this repository.

## Input contract

- Clinicopathologic variables: `schemas/clinical_variables.csv`
- Radiomic-spatial feature names: `schemas/radiomics_spatial_features.txt`
- CNN embedding feature names: `schemas/cnn_features.txt`
- Complete machine-readable schema: `schemas/input_schema.json`

Missing continuous values are median-imputed using the frozen development-cohort preprocessing parameters. Categorical levels not observed during development are ignored by the frozen one-hot encoder. Regional nodal metastasis, adjacent-organ invasion, and metastatic-node count are **not inputs** to this release; examined lymph-node count is retained. Imaging features must be generated using the study preprocessing and feature-extraction definitions described in the manuscript and supplemental methods.

The handcrafted representation contains PyRadiomics features extracted after isotropic 1.0-mm resampling and a separately engineered spatial/context block computed from isotropic 2.5-mm cached arrays. The CNN encoder receives four channels (CT, bowel wall, primary tumor, and regional nodes) and returns a 128-dimensional embedding. These branches are kept separate because they reproduce the frozen analysis pipeline.

## Reproducibility and privacy

The originally serialized lifelines object retained individual development-cohort follow-up times, event indicators, and fitted partial hazards. Those fields are unnecessary for inference and are therefore excluded from this release. The public package retains only frozen preprocessing objects, model coefficients, normalization parameters, and baseline cumulative hazards at the prespecified prediction horizons. The latest model is exported from `M12_latest_clinical.joblib`; verification results and package hashes are recorded in `docs/VERIFICATION_REPORT.md` and `release_manifest.json`.

See `DATA_ACCESS.md` for the controlled-access procedure and `MODEL_CARD.md` for limitations.

## Citation and permanent archive

For peer review, replace the placeholders below after creating the immutable release:

- Repository: `[PUBLIC_REPOSITORY_URL]`
- Release: `v1.1.0`
- Commit: `[COMMIT_SHA]`
- Archived DOI: `[ZENODO_DOI]`

The repository and archived release should be retained for at least five years after publication, consistent with RSNA policy.

## License

Code is released under the MIT License. Model parameters are supplied for noncommercial research use under the terms in `MODEL_LICENSE.txt`. No rights are granted to the underlying patient data, CT images, or segmentation masks.
