# Fall Risk Date-Independent Model (6M Rollover)

This folder documents and packages the **date-independent fall-risk pipeline** using three scripts:

1. `fall_risk_feature_engineering.py`  
   Generates engineered features from wide/long monthly source data, with optional target creation.
2. `fall_risk_training_6m.py`  
   Trains and validates a 6-month model and saves a `.pkl` artifact.
3. `fall_Risk_score_6m.py`  
   Loads the trained model and scores next-month member fall risk.

---

## Pipeline Overview

### Step 1: Feature Engineering

- Input: `202411_to_202602_base_features.parquet` (or csv/xlsx/xls supported by script)
- Detects monthly columns dynamically and builds rolling-window features
- Outputs separate window files for fair model comparison:
  - `fall_risk_features_last_6m.parquet`
  - `fall_risk_features_last_4m.parquet`
  - `fall_risk_features_last_3m.parquet`
- Optional target column:
  - `target_fall_next_month` (for training dataset)

### Step 2: Training (6M)

- Input: engineered **6m** feature file with target
- Splits labeled data with stratified validation
- Trains via `FallRiskForecastingModel` (from `FallRisk_Healthplans_Incremental.py`)
- Selects validation threshold with FPR/FNR caps
- Saves:
  - model `.pkl`
  - validation plots and summary artifacts

### Step 3: Scoring (6M)

- Input: engineered **6m** feature file without target + model `.pkl`
- Produces compact report:
  - `account_number`
  - `Risk_Score` (1..10)
  - `Risk_Category` (Low/Medium/High)

---

## Recommended Project Layout

```text
fall_risk_date_independent_model/
  fall_risk_feature_engineering.py
  fall_risk_training_6m.py
  fall_Risk_score_6m.py
  FallRisk_Healthplans_Incremental.py
  requirements.txt
  .gitignore.txt
  GETTING_STARTED.md
  README.md
  outputs/
```

---

## Key Run Commands

See `GETTING_STARTED.md` for full commands. Typical sequence:

1. Generate features with target (`--include_target`)
2. Train model from `fall_risk_features_last_6m_with_target.parquet`
3. Generate features without target
4. Score using `fall_Risk_score_6m.py`

---

## Notes

- Keep training and scoring feature generation separate to avoid label leakage.
- Use consistent `predict_month` between model thresholding and score-time threshold lookup.
- If file names are provided without extensions, scripts may auto-detect parquet/csv variants.
- Keep large data/model artifacts out of source control (`.gitignore.txt` covers this).
