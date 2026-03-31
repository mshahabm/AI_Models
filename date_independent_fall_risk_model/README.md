# Fall Risk Modeling Pipeline (12-Month Rollover)

## Overview
This project implements an end-to-end fall risk prediction pipeline that transforms raw longitudinal member data into actionable risk scores.
The workflow is designed to handle real-world healthcare data challenges such as temporal variability, missing values, and class imbalance.
The pipeline consists of three core stages:
* Feature Engineering – Transform raw monthly data into structured temporal features
* Model Training & Validation – Train a supervised model to predict next-month fall risk
* Inference & Scoring – Generate risk scores and categories for new data

### High-Level Workflow
Raw Monthly Data → Feature Engineering (12-Month Window) → Model Training → Saved Model → Inference → Risk Scores & Categories

---

## 1. Feature Engineering (fall_risk_feature_engineering.py)

### Purpose
Transforms raw member-level data into model-ready features using a rolling time window (e.g., last 12 months).

### Key Capabilities
* **Flexible Input Handling**
* Supports .parquet, .csv, .xlsx
* Automatically detects:
* Wide format (metric_MM_YYYY)
* Long format (obs_month) and converts to wide
* **Temporal Windowing**
* Builds features over configurable windows (e.g., 9, 12 months)
* Uses latest month (or custom anchor) as reference point
* **Feature Types**
* Event Trends (e.g., fall_count trend)
* Time-Lag Features (T-1, T-2, … per metric)
* Mobility Signals (steps trend, low activity ratio)
* Sentiment Signals (positive/negative/neutral counts)
* Data Quality Metrics (completeness, sparsity)
* Behavioral Activity Patterns
* **Target Generation (Optional)**
* Creates target_fall_next_month based on next month's fall count

---

## 2. Model Training (fall_risk_training_12m.py)

### Purpose
Trains a supervised model to predict probability of a fall in the next month.

### Key Components
* **Automatic Feature Selection**
* Excludes identifiers and target column
* **Data Preparation**
* Drops rows with missing target values (unlabeled data)
* Ensures binary target (0/1)
* **Train / Validation Split**
* Uses Stratified Shuffle Split to preserve class balance
* **Model Options**
* XGBoost (default)
* Random Forest
* Gradient Boosting
* Logistic Regression
* **Adaptive Threshold Optimization**
* Selects optimal decision threshold based on:
* False Positive Rate (FPR)
* False Negative Rate (FNR)
* Uses weighted tradeoff to avoid extreme bias
* **Evaluation Metrics**
* ROC-AUC
* Precision / Recall
* Confusion Matrix
* Threshold diagnostics
* **Outputs**
* Trained model: fall_risk_model_6m.pkl
* Validation artifacts:
* ROC curve
* Confusion matrix (image + text)
* Performance report
* Training metadata

---

## 3. Inference & Scoring (fall_Risk_score_12m.py)

### Purpose
Applies the trained model to new data and generates interpretable risk scores.

### Key Features
* **Model Loading**
* Loads pre-trained model artifact (.pkl)
* **Probability Prediction**
* Predicts fall probability for each member
* **Adaptive Thresholding**
* Uses month-specific threshold from training
* **Score Mapping (1–10 Scale)**
* Converts probabilities into risk scores using percentile ranking:
* Low Risk: 1–2
* Medium Risk: 3–6
* High Risk: 7–10
* **Risk Categorization**
* Maps numeric score to:
* Low
* Medium
* High
* **Outputs**
* Final report:

---

## Key Design Principles
* **✅ Temporal Awareness**
* Uses rolling windows to capture recent behavioral patterns
* **✅ Robust Feature Engineering**
* Combines:
* Trends
* Recency signals
* Activity levels
* Data completeness
* **✅ Controlled Decision Thresholds**
* Explicitly balances false positives vs false negatives
* **✅ Separation of Concerns**
* Feature engineering, training, and inference are modular
* **✅ Production-Ready**
* Supports:
* Multiple file formats
* Automated file detection
* Scalable pipelines

---

## Summary
This pipeline provides a robust, scalable, and interpretable framework for predicting fall risk using longitudinal data.
By combining temporal feature engineering, controlled model training, and adaptive scoring, it delivers reliable risk stratification suitable for real-world healthcare applications.
