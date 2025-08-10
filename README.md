# Employee Attrition Prediction: Explainable Machine Learning Report

This project applies interpretable machine learning techniques to predict employee attrition using logistic regression, with a focus on transparency and stakeholder communication. Both **global** and **local** model explanations are included, leveraging model coefficients and SHAP values.

---

## Objective

Predict which employees are at risk of leaving the company and understand *why*, using:

- **Logistic Regression** for interpretability.
- **SHAP Values** for individualized explanations and global patterns.
- A structured, end-to-end ML pipeline with preprocessing and evaluation.

---

## 1. Logistic Regression Coefficients (Global Explainability)

Logistic regression provides a direct mapping between feature values and their contribution to the log-odds of attrition.

- Positive coefficients increase attrition risk.
- Negative coefficients reduce it.

### Key Highlights

- **Strongest Positive Predictors**:  
  `EducationField_Technical Degree`, `JobRole_Research Scientist`, and `BusinessTravel_Non-Travel`.

- **Strongest Negative Predictors**:  
  `JobRole_Healthcare Representative`, `JobRole_Manager`, and `BusinessTravel_Travel_Rarely`.

- **Minimal Impact Features**:  
  Features like `Gender`, `Education`, and `JobRole_Human Resources` showed negligible coefficient values.

### [Insert Coefficients Bar Chart Here]

---

## 2. SHAP Value Interpretation (Global + Local Explainability)

SHAP values explain predictions on a per-observation basis and provide model-agnostic insights.

- Each dot shows how a feature contributed to an individual prediction.
- Color indicates feature value (blue = low, red = high).
- Horizontal position reflects direction/magnitude of influence.

### Key Global Insights (SHAP)

- **Most Influential Features**:  
  `NumCompaniesWorked`, `TotalWorkingYears`, `YearsWithCurrManager`, and `EnvironmentSatisfaction`.

- **Contrast With Coefficients**:  
  Some features with low coefficients (e.g. `NumCompaniesWorked`) had high SHAP impact—emphasizing their interaction effects or conditional relevance.

### [Insert SHAP Summary Plot Here]  
### [Insert SHAP Beeswarm Plot Here]

---

## 3. Final Thoughts

- Coefficients tell us how the model is built.
- SHAP tells us how the model behaves.
- Together, they give a full picture: the "rules" (coefficients) and the "realities" (SHAP).

This project demonstrates the value of combining linear interpretability with model-agnostic explanation tools to surface actionable insights in HR analytics.

---

## 🔧 Repository Structure

```
.
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── final_model_pipeline.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_explainability.ipynb
│   └── 05_final_report.ipynb
└── README.md
```

---

## 🔍 Next Steps

- Add SHAP-based cohort profiling for team-level analysis.
- Implement visual dashboards using Streamlit or Tableau.
- Extend analysis with Random Forest or XGBoost for performance benchmarking.

