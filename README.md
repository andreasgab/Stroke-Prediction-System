# Stroke Prediction System
An end-to-end Machine Learning pipeline utilizing AdaBoost and ADASYN to predict stroke risk. Optimized for medical clinical priority using F2-Scoring to maximize patient recall (88% on unseen data). Includes a SHAP-based explainability layer and a Streamlit deployment dashboard.

# Project Overview
This system is designed to assist healthcare professionals in identifying high-risk stroke patients. Unlike standard models that prioritize overall accuracy, this system is specifically tuned for **Recall**. In a clinical screening context, the cost of a "False Negative" (missing a stroke) is far higher than a "False Positive" (further testing for a healthy patient).

## Key Features

* **Medical-First Optimization:** Uses the F2-Score (β=2) as the primary evaluation metric to prioritize Recall over Precision.
* **Imbalanced Data Handling:** Implements ADASYN (Adaptive Synthetic Sampling) to address the significant class imbalance in stroke occurrences.
* **Dual-Model Deployment:** The application features a model-switcher to compare a Validated Model (4k training rows) against a Full Knowledge Model (5k training rows).
* **Automated Scikit-Learn Pipeline:** Uses a ColumnTransformer to handle median imputation for BMI and One-Hot Encoding for categorical features automatically.
* **SHAP Explainability:** Provides a "Glass Box" view of predictions, showing exactly which features (like Age or Glucose) pushed a specific patient toward a high-risk score.
* **mRMR Integration:** Utilized Minimum Redundancy Maximum Relevance feature selection to ensure clinical markers (Hypertension/Heart Disease) contribute to the model despite the mathematical dominance of Age.

## Performance Summary
* Evaluated against a 1,000-row "Production Simulation" dataset (unseen by the Validated model):
    Recall: 87% (Successfully flagged 53 out of 61 actual stroke cases).
    Precision: 15% (A deliberate trade-off to ensure a high-sensitivity screening tool).
    Primary Predictors: Age (Dominant), BMI, Glucose Level, and Marital Status.

* Evaluated against the full 5,000-row dataset:
    Recall: 88% (218/249 stroke cases).
    Precision: 12%.
    The lower precision is a direct result of the ADASYN-oversampling and F2-optimization, ensuring that the model errs on the side of caution.

## Model Interpretability (SHAP Insights):
* **The Age Factor:** Age is the dominant driver of risk. SHAP analysis identifies a significant "risk escalation" point between 50–60 years old.
* **Proxy Correlations:** The model leverages Marital Status (ever_married) as a strong proxy for life stage and age, reinforcing risk signals for older demographics.

## Project Structure

├── model/                          # Serialized .pkl pipelines (Preprocessor + AdaBoost)

|  ├── adaboost_stroke_prediction_4k_trained_model_f2.pkl

|  └── adaboost_stroke_prediction_5k_trained_model_f2.pkl

├── data/                           # Raw dataset and production simulation CSVs

├── analysis.ipynb                  # Full development: Preprocessing, model selection, GridSearch, Gini importance, SHAP analysis, MRMR

├── requirements.txt                # Project dependencies

└── stroke_prediction_app.py        # Web-app interactive Streamlit dashboard. Open terminal and enter " streamlit run "stroke_prediction_app.py" "