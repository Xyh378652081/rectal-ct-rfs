# Model Card: M12 Full-Fusion Recurrence-Risk Model

## Intended use

M12 is a research model for estimating recurrence-free survival risk after curative-intent surgery for primary rectal cancer using preoperative portal-venous phase contrast-enhanced CT-derived features and postoperative clinicopathologic variables. It is intended for methodological reproduction and external research validation.

## Not intended for

- direct patient care or autonomous treatment selection;
- estimation of treatment benefit;
- use in populations or imaging protocols without local validation;
- reconstruction or inference of protected patient information.

## Model components

The frozen model combines 13 clinicopathologic variables (23 encoded features), 20 principal components from multistructure radiomic-spatial features, and 12 principal components from a frozen self-supervised 3D CNN representation. A penalized Cox proportional hazards model uses a 55-feature design and produces relative risk and event-risk estimates at 12, 24, and 36 months. The examined lymph-node count is retained; regional nodal metastasis, adjacent-organ invasion, and metastatic-node count were removed in the latest update.

## Development and validation

The model was developed at one center (n=796) with patient-level out-of-fold internal evaluation of downstream modeling conditional on frozen image representations and assessed in three geographic external centers (pooled n=262). In the locked analysis, which used center-specific censoring distributions and center-stratified bootstrap resampling, the pooled external 36-month AUC was 0.917 (95% CI, 0.866–0.954), the external C-index was 0.819, and the 36-month Brier score was 0.175. The external centers had been inspected during previous analysis iterations and are not an untouched prospective test set.

## Limitations

The study was retrospective. The external sample included 79 events, limiting the precision of center-specific calibration. No independent repeat segmentation experiment was performed. The model was not directly compared with an MRI-based model and has not undergone prospective workflow, reader-impact, or model-guided intervention evaluation. The self-supervised encoder was not retrained within every outer fold. External calibration showed absolute-risk overestimation; the released event-risk probabilities must not be used as treatment or surveillance thresholds without independent validation and recalibration.

## Fairness and subgroup considerations

The released model has not been established as equitable across all demographic, geographic, scanner, treatment, or disease-stage subgroups. Users should assess calibration and performance in the intended population and report subgroup results with uncertainty.

## Privacy

The public artifacts exclude patient identifiers, patient-level outcomes, follow-up times, predictions, images, and labels. Model parameters may still encode statistical information about the development cohort and must not be used for membership-inference or re-identification attempts.
