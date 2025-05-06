# Fairness Analysis of the COMPAS Recidivism Prediction Model

## Abstract

This repository presents a comprehensive evaluation of a logistic regression model trained to predict two-year recidivism using the COMPAS dataset. Beyond standard performance metrics, we assess group fairness across gender and age cohorts by computing Statistical Parity, True Positive Rate, False Positive Rate, and Positive Predictive Value for each subgroup. The results reveal notable disparities, underscoring the need for fairness-aware interventions in algorithmic risk assessments.

## Introduction

Recidivism risk assessments like COMPAS increasingly inform high-stakes criminal justice decisions, yet they may perpetuate or amplify existing societal biases. This analysis documents a transparent, interpretable pipeline and quantifies fairness across protected attributes (gender and age), highlighting where disparities emerge.

## Data Description

The COMPAS dataset (ProPublica) includes demographic features, criminal history, and observed recidivism within two years of arrest. After applying ProPublica’s recommended filters, we focus on:

* **age**
* **sex** (binary flag)
* **priors\_count** (number of past offenses)
* **c\_charge\_degree** (charge severity)
* **two\_year\_recid** (binary target)

## Data Preprocessing

1. **Filtering:** Valid cases only (screening within ±30 days of arrest, known outcomes, felony charges).
2. **Encoding:** Converted `sex` to `sex_binary` (0 = Female, 1 = Male) and one-hot encoded `c_charge_degree`.
3. **Binning:** Grouped `age` into buckets: `< 25`, `25–35`, `35–45`, `> 45`.
4. **Split:** 70% train, 30% test (random\_state=42).

## Model Training

Trained a logistic regression classifier (`scikit-learn`, `max_iter=1000`) on four predictors: `age`, `sex_binary`, `priors_count`, and one-hot `c_charge_degree`.

## Performance Evaluation

* **Accuracy:** 68.0%
* **Precision:** 66.0%
* **Recall (TPR):** 57.0%

## Fairness Analysis

### Gender-Based Metrics

| Metric                    | Female | Male   | Disparity (pp) |
| ------------------------- | ------ | ------ | -------------- |
| Statistical Parity (SR)   | 9.35%  | 46.03% | 36.68          |
| True Positive Rate (TPR)  | 19.85% | 64.34% | 44.49          |
| False Positive Rate (FPR) | 3.15%  | 29.95% | 26.80          |
| Positive Predictive Value | 78.79% | 65.36% | 13.43          |

### Age-Based Metrics

| Age Group | SR     | TPR    | FPR    | PPV    | Disparity (SR) |
| --------- | ------ | ------ | ------ | ------ | -------------- |
| < 25      | 57.40% | 69.40% | 43.53% | 64.81% | 42.13          |
| 25–35     | 42.18% | 59.48% | 23.94% | 72.38% |                |
| 35–45     | 29.66% | 46.96% | 20.28% | 55.67% |                |
| > 45      | 15.27% | 29.70% | 9.35%  | 56.60% |                |

## Discussion

While overall predictive performance is acceptable for exploratory purposes, significant disparities—particularly in True Positive and False Positive Rates—highlight fairness concerns. The model not only reflects underlying data imbalances but exacerbates them, disproportionately impacting men and younger defendants.

**Mitigation Strategies**

* Reweigh training samples to balance distributions.
* Adversarial debiasing during model training.
* Post-processing adjustments (e.g., equalized odds).

## Conclusion

This pipeline demonstrates a full data-science lifecycle enriched with fairness auditing. Despite reasonable accuracy, the model introduces large demographic gaps that could undermine trust and justice if deployed. Future work should prioritize fairness-aware methods to produce more equitable risk assessments.

## References

* ProPublica COMPAS dataset: ProPublica. “COMPAS Recidivism Risk Score Data and Analysis.” 2016.
