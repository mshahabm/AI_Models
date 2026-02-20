# Fall Risk Forecasting — DTC (Direct-to-Consumer)

Incremental fall risk forecasting pipeline for member-level predictions for DTC members. Uses historical feature data (e.g. fall counts, daily steps, operator notes) to predict fall risk for the next month and outputs risk scores, probabilities, and evaluation metrics.

---

## Overview

- **Script**: `fall_risk_DTC.py`
- **Mode**: Incremental — process one new month at a time (~10–15 min per month; ~90% cost savings vs full retrain).
- **Input**: Training file (and optional test file) in CSV or Parquet with monthly feature columns (`<feature>_MM_YYYY`).
- **Output**: Per-member risk scores and probabilities; validation/test ROC-AUC, confusion matrices, feature importance; optional updated training file for the next run.

---

## Features

- **Algorithms**: XGBoost (default), RandomForest, GradientBoosting, LogisticRegression.
- **Feature handling**: Month N features → predict fall in month N+1; temporal steps features; NaN-aware engineered features (e.g. data quality score, age×steps).
- **Calibration**: Probability calibration (e.g. isotonic) and adaptive thresholds per month.
- **Stratified split**: 80/20 train/validation by account, stratified by fall status.
- **I/O**: CSV or Parquet; output format matches input. Optional pyarrow for Parquet.

---

## Quick Start

1. **Setup**  
   Install: `pandas`, `numpy`, `scikit-learn`, `xgboost`. Optional: `pyarrow`, `matplotlib`, `imbalanced-learn`, `seaborn`.

2. **Data**  
   Place a training file named like `FallRisk_Training_<StartMMYYYY>_To_<EndMMYYYY>_DTC.parquet` (or `.csv`) in the project directory. Optionally add a test file: `FallRisk_Test_<MM_YYYY>_DTC.parquet`.

3. **Run (manual, one month)**  
   ```bash
   python fall_risk_DTC.py --predict_month 05_2025 --training_file FallRisk_Training_022024_To_042025_DTC
   ```

4. **Run (automated)**  
   ```bash
   python fall_risk_DTC.py --auto
   ```

For detailed setup, file naming, and workflow, see **[Getting_Started.md](Getting_Started.md)**.

---

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--predict_month MM_YYYY` | Month to predict (required in manual mode). |
| `--training_file <path>` | Training file path (required in manual mode). Extension optional. |
| `--test_file <path>` | Optional test file for evaluation and training update. |
| `--algorithm` | `XGBoost` (default), `RandomForest`, `GradientBoosting`, `LogisticRegression`. |
| `--auto` | Automated mode: find latest training file, process next month(s) until no test file or max iterations. |

---

## Outputs

| Output | Description |
|--------|-------------|
| `FallRisk_BetaVersion_Predict_<MMYYYY>/` | Predictions and validation metrics (ROC, CM, performance, feature importance). |
| `Fall_Risk_score_<MMYYYY>.parquet` or `.csv` | One row per member: `account_number`, `Probability`, `Risk_Score`, `Risk_Category`, `Flagged`, `Data_Quality_Pct`, etc. |
| `FallRisk_BetaVersion_Test_Including_<MMYYYY>/` | Test ROC, CM, and performance (when test file is provided). |
| `FallRisk_Training_<Start>_To_<NewEnd>_DTC.<ext>` | Updated training file including the predicted month (when test file is provided). Use as next `--training_file`. |

---

## Related Scripts

- **EDA_DTC.py** — Exploratory data analysis: total unique accounts and % of accounts with data per feature (from `beta_health_plan_list.CSV` or another CSV/Parquet). Run: `python EDA_DTC.py` or `python EDA_DTC.py <filename>`.

---

## Data Assumptions

- **Account column**: `account_number`.
- **Target**: `fall_count_<MM_YYYY>` (or `fall_alarm_count_<MM_YYYY>` as fallback) > 0 → positive class.
- **Months** supported in code: `11_2024` through `02_2026`; extend `month_order` in `fall_risk_DTC.py` if needed.
