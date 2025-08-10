# Final Report

This report turns our HR data into clear, actionable guidance for reducing turnover (attrition)—showing which employees are most likely to leave and why. The model consistently points to frequent travel, overtime, and short tenure as the biggest warning signs.

---

## Executive Summary

- **Final model:** Logistic Regression (with SMOTE on train), tuned via 5-fold GridSearch.  
- **Operating threshold:** 0.79 (selected to maximize F1 on validation).  
- **Validation (thr=0.79):** AUC 0.827, Accuracy 0.874, Precision 0.632, Recall 0.511, F1 0.565.  
- **Test (thr=0.79):** AUC 0.808, Accuracy 0.861, Precision 0.571, Recall 0.511, F1 0.539.  
- **Key drivers (global):** frequent travel, overtime, short tenure; protective signals include long tenure, life sciences/medical education, and managerial roles.  

![alt text](../exports/figures_1/ROC_curve_test.png)

---

## 1) Data & EDA

- **Dataset shape:** 1,470 rows × 35 columns originally; after dropping non-informative columns, **31 columns** remain.  
- **Target distribution:** 83.88% *No*, 16.12% *Yes* (significant class imbalance).  
- **Notable patterns:** Overtime employees are nearly **3×** more likely to leave; higher attrition among younger staff, longer commutes, lower income/stock options, and certain roles (Sales, Laboratory Technician).  
- **Correlations:** Compensation & tenure metrics cluster strongly (e.g., JobLevel ↔ MonthlyIncome), while satisfaction metrics are largely independent.

![alt text](../exports/figures_1/02_class_distribution.png)
![alt text](../exports/figures_1/03_correlation_heatmap.png)

---

## 2) Preprocessing & Feature Engineering 

**Pipeline highlights** (exported as `models/preprocessing_pipeline.pkl`):

- **Feature engineering** (subset): `TenureCategory`, `TenureGap`, `TenureRatio`, `ZeroCompanyTenureFlag`, `NewJoinerFlag`, `OverTime_JobLevel`, `Travel_Occupation`, `SatisfactionMean`, `SatisfactionRange`, `SatisfactionStability`, `Log_MonthlyIncome`, `Log_DistanceFromHome`, `LowIncomeFlag`, `StressRisk`.
- **Transforms:** One-hot encode key categoricals; standardize numeric features; pass through engineered binary flags.
- **Leakage control:** Fit pipeline on **train only**; apply consistently across splits.

---

## 3) Modeling 

**Data split:** 60/20/20 (train/validation/test), stratified. After transformation: **Train (882 × 75)**, **Val (294 × 75)**, **Test (294 × 75)**.  
**Resampling:** SMOTE on **train** only.  
**Search space:** `C ∈ {0.01, 0.1, 1, 10}`, penalties `l1|l2`, `class_weight ∈ {None, "balanced"}`; **best CV AUC ≈ 0.896**.  
**Validation ROC AUC:** **0.827**; threshold tuning across 0.01–0.99 → **best threshold = 0.79** (max F1).

**Validation metrics @ 0.79**  
- Precision 0.632 | Recall 0.511 | F1 0.565 | Accuracy 0.874 | AUC 0.827

**Test metrics @ 0.79**  
- Precision 0.571 | Recall 0.511 | F1 0.539 | Accuracy 0.861 | AUC 0.808

![](../notebooks/04_interpretation.ipynb)
![alt text](../exports/figures_1/05_validation_confusion_matrix.png)
![alt text](../exports/figures_1/08_confusion_matrix_05.png)
![alt text](../exports/figures_1/07_confusion_matrix_test.png)

---

## 4) Interpretation

**Global drivers (SHAP):**  
- **Risk ↑**: Frequent travel (esp. Sales roles), Overtime, Short tenure (≤3 yrs), more prior companies.  
- **Risk ↓**: Long tenure/total working years, higher satisfaction, stock options/raises, life sciences/medical education, managerial roles; older age modestly protective.  

**Model internals:** Logistic regression **intercept ≈ -3.606**. Coefficient view aligns with SHAP: travel intensity and workload amplify risk; tenure/skills/management stability reduce it.

Suggested visuals:
![alt text](../exports/figures_1/09_beeswarm.png)
![alt text](../exports/figures_1/10_waterfall_high_risk.png)
![alt text](../exports/figures_1/11_waterfall_borderline.png)
![alt text](<../exports/figures_1/12_waterfall _low_risk.png>)
![alt text](../exports/figures_1/13_top_coef.png)

---

## 5) Business Implications & Recommendations

1) **Target high-burnout segments:** Sales roles with **frequent travel** and **overtime**.  
2) **Tenure-building:** Reduce early exits (≤3 yrs) via onboarding, mentorship, and defined promotion paths.  
3) **Incentives:** Maintain competitive raises/stock options to anchor mid-tenure staff.  
4) **Commute-aware flexibility:** Where feasible, remote/hybrid options for long-commute employees.

---

## 6) Risks & Notes

- Synthetic dataset; findings should be validated against real HR data before policy changes.  
- Imbalance remains; threshold and cost-sensitive tuning should be revisited per business tolerance for false positives/negatives.  
- Monitor for **fairness** across demographics during deployment (post-hoc checks and periodic audits).

---

## 7) Next Steps

- Evaluate alternative models (e.g., gradient boosting) with **calibration** and **cost curves**.  
- Pilot interventions for high-risk cohorts; A/B test impact on attrition rates.  
- Build a lightweight scoring & explanation service using the saved pipeline + model.

---
