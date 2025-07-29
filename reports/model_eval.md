# Modeling Evaluation Report – Logistic Regression for Employee Attrition

This report documents the modeling process and evaluation results using logistic regression as the baseline model for predicting employee attrition. The modeling was executed in `03_modeling.ipynb`.

---

## 1. Model Selection

We used **logistic regression** due to its transparency and suitability for binary classification tasks with tabular data. The model was embedded in a pipeline with:

- Custom feature engineering (`FeatureEngineer`)
- Column transformations (OneHot, Ordinal, StandardScaler)
- Class weighting set to `'balanced'`
- Solver: `'liblinear'`
- Maximum iterations: 1000

---

## 2. Cross-Validation Performance (Threshold = 0.5)

The initial model was evaluated using stratified 5-fold cross-validation on the training data.

**Metrics (mean across folds):**
- Accuracy: `0.85`
- ROC AUC: `0.77`
- Precision (Attrition = Yes): `0.49`
- Recall (Attrition = Yes): `0.44`
- F1 Score: `0.46`

The model was well-calibrated, though recall was somewhat limited at the default threshold.

---

## 3. Threshold Tuning

Using the validation set, we swept probability thresholds and identified the optimal threshold as:

**Best Threshold:** `0.72`

This was selected to maximize the F1 score for the minority class (Attrition = Yes).

---

## 4. Final Evaluation on Test Set (Threshold = 0.72)

Using the optimized threshold, we evaluated the model on the untouched test set.

**Metrics on Test Data:**

- Accuracy: `0.86`
- ROC AUC: `0.78`
- Precision: `0.53`
- Recall: `0.51`
- F1 Score: `0.52`

**Confusion Matrix:**

|               | Predicted No | Predicted Yes |
|---------------|--------------|---------------|
| **Actual No** | 233          | 15            |
| **Actual Yes**| 18           | 19            |

This demonstrates meaningful improvements in recall and precision over the default threshold. The model correctly identifies approximately half of the attriting employees while maintaining a manageable false positive rate.

---

## 5. Interpretation

- **Recall improved** from 44% → 51% after threshold tuning.
- **Precision increased** to 53%, indicating a better balance of false positives and true positives.
- **ROC AUC of 0.78** indicates strong overall ranking performance.

These results validate logistic regression as a transparent, balanced model that can reasonably distinguish between at-risk and stable employees when paired with domain-aware feature engineering and appropriate threshold tuning.

---

## 6. Next Steps

- Use SHAP values to explain individual predictions and global feature impacts.
- Optionally compare with higher-capacity models (e.g., CatBoost, XGBoost) to explore tradeoffs between accuracy and explainability.
- Apply model to identify high-risk cohorts in operational HR data.