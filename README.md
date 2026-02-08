# Stroke-Prediction-System
An end-to-end Machine Learning pipeline utilizing AdaBoost and ADASYN to predict stroke risk. Optimized for medical clinical priority using F2-Scoring to maximize patient recall (87% recall on unseen data). Includes a SHAP-based explainability layer.

# Stroke Prediction & Risk Analysis
This project implements a machine learning system designed to assist healthcare professionals in identifying high-risk stroke patients. Unlike standard models that prioritize overall accuracy, this system is specifically tuned for **Recall**, ensuring that the highest possible number of potential stroke cases are flagged for clinical review.

## Key Features
* **Medical-First Optimization:** Uses an **F2-Score** metric to prioritize Recall over Precision (Beta=2).
* **Imbalanced Data Handling:** Implements **ADASYN** (Adaptive Synthetic Sampling) to handle the significant class imbalance in stroke occurrences.
* **Automated Pipeline:** Full Scikit-Learn pipeline integrating `ColumnTransformer`, `StandardScaler`, and `AdaBoost`.
* **Model Explainability:** Uses **Gini Importance** and **SHAP (SHapley Additive exPlanations)** to visualize feature impact, making the "Black Box" model transparent for clinical use.

## Performance Summary
After rigorous Cross-Validation and testing on a 1,000-row "Production Simulation" dataset, the model achieved:
* **Recall:** 87% (Caught 53 out of 61 actual strokes)
* **Precision:** 15% (Acceptable trade-off for life-saving screening)
* **Top Predictors:** Age, BMI, Smoking Status Yes/No and Working Environment.


## Project Structure
```text
├── data/                         # Raw and simulated production CSV files
├── saved_model/                  # Trained .pkl pipeline (Best Estimator)
├── analysis.ipynb                # Full development, GridSearch, and SHAP analysis
├── requirements.txt              # Necessary libraries (scikit-learn, imbalanced-learn, shap, etc.)
└── streamlit_app.py              # (Optional) Interactive dashboard code