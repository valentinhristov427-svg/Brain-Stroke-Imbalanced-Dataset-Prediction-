# Brain Stroke Prediction — Capstone Project

A classification project predicting stroke occurrence on an imbalanced dataset (~5% positive class), built as part of the Rockborne Data Analytics programme.

## Overview

The dataset (`brain_stroke_data.csv`) contains patient demographic and health features, with `stroke` as a binary target. Given the ~19:1 class imbalance, the project focuses on F1 (stroke class) as the evaluation metric rather than accuracy, which would be misleadingly high for a majority-class classifier.

## Process

1. **Data quality checks** confirmed no missing values, no duplicates, clean categorical values, and physiologically valid numeric ranges (including legitimate fractional ages for infants).
2. **EDA** identified `age` as the dominant predictor, a bimodal relationship between `avg_glucose_level` and stroke risk, and several categorical features (`ever_married`, `work_type`, `smoking_status`) substantially confounded with age.
3. **Preprocessing** built as separate `sklearn` pipelines per model: one-hot encoding for categoricals throughout, with `StandardScaler` added for the logistic regression pipeline (not needed for tree-based models).
4. **Model comparison** Logistic Regression and XGBoost were compared via stratified 5-fold cross-validation, with hyperparameter tuning (`GridSearchCV`) and decision threshold tuning (via `precision_recall_curve`) applied to both.
5. **Imbalance handling** `class_weight='balanced'` (Logistic Regression) and `scale_pos_weight` (XGBoost) were used as the primary strategies; SMOTE and random undersampling were also tested for XGBoost but underperformed.

## Results

| Model | Threshold | F1 | Precision | Recall |
|---|---|---|---|---|
| **Logistic Regression** | 0.745 | **0.277** | 0.188 | 0.529 |
| XGBoost (tuned) | 0.617 | 0.261 | 0.182 | 0.466 |

Logistic Regression was selected as the final model, outperforming XGBoost despite the latter's theoretical advantage in capturing non-linear structure. This is attributed to age's dominant, near-linear relationship with stroke risk and the small positive class size (~224 cases), which favors a simpler, better-regularized model over higher-capacity alternatives.

## Repository structure

- `Project_Brain_Stroke_Capstone.ipynb` — full notebook: data quality checks, EDA, preprocessing, model comparison, and final prediction pipeline.
- `Data` 
  - `brain_stroke_data.csv` — labelled training data.
  - `brain_stroke_unseen.csv` — unseen data for prediction.
- `Predictions` 
  - `Valentin_Hristov_brain_stroke.csv` — final submission (single column `Predictions`, one row per unseen record).

## Key takeaways

- Threshold tuning had a larger impact on F1 than hyperparameter tuning for both models.
- The simplest model (Logistic Regression) outperformed more complex alternatives.

