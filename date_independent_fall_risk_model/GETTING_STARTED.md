# Getting Started

This quick guide runs the complete **date-independent 12-month rollover** fall-risk pipeline with:

1. `fall_risk_feature_engineering.py`
2. `fall_risk_training_12m.py`
3. `fall_Risk_score_12m.py`

Two possible workflows:
1) Train the model for the first time and then score
fall_risk_feature_engineering.py with targets -> fall_risk_training_12m.py -> fall_risk_feature_engineering.py without targets -> fall_Risk_score_12m.py
2) Used already trained model to score
all_risk_feature_engineering.py without targets -> fall_Risk_score_12m.py
							

## 1) Folder Setup

Put these files into `fall_risk_date_independent_model`:

- `fall_risk_feature_engineering.py`
- `fall_risk_training_12m.py`
- `fall_Risk_score_12m.py`
- `FallRisk_Healthplans_Incremental.py` (shared model utilities imported by training/scoring)
- `requirements.txt`

Input data file (example):

- `202411_to_202602_base_features.parquet`


## 2) Create Environment

```powershell
cd fall_risk_date_independent_model
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3) Generate Engineered Features

### 3.1 Training features (with target)
Creates 1 file for windows 12m. 
For 12m model training, use the `fall_risk_features_last_12m_with_target.parquet` output.

```powershell
python fall_risk_feature_engineering.py `
  --input_file "202411_to_202602_base_features.parquet" `
  --output_dir "outputs\features_with_target" `
  --windows 12 `
  --include_target
  -- anchor_month 01_2026
```

### 3.2 Scoring features (without target)
Use this for production inference/scoring input.

```powershell
python fall_risk_feature_engineering.py `
  --input_file "202411_to_202602_base_features.parquet" `
  --output_dir "outputs\features_for_scoring" `
  --windows 12
```

## 4) Train 12-Month Model

```powershell
python fall_risk_training_6m.py `
  --train_file "outputs\features_with_target\fall_risk_feature_last_12m.parquet" `
  --target_col "target_fall_next_month" `
  --predict_month "02_2026" `
  --output_dir "outputs\training_12m" `
  --model_file "outputs\training_12m\fall_risk_model_12m.pkl"
```

## 5) Score Next-Month Fall Risk

```powershell
python fall_Risk_score_12m.py `
  --feature_file "outputs\features_for_scoring\fall_risk_feature_last_12m.parquet" `
  --model_file "outputs\training_12m\fall_risk_model_12m.pkl" `
  --predict_month "03_2026" `
  --output_file "outputs\scoring\Fall_Risk_score_032026_12m.parquet"
```

## 6) Expected Outputs

- Feature files:
  - `fall_risk_features_last_12m.parquet`
  - `fall_risk_features_last_12m.parquet` (when `--include_target`)
- Trained model:
  - `fall_risk_model_12m.pkl`
- Training diagnostics:
  - ROC/CM plots, metrics text, metadata CSV
- Scoring report:
  - `account_number`, `Risk_Score`, `Risk_Category`

## 7) Common Checks

- Ensure `target_fall_next_month` exists (or is derivable) in training feature file.
- Ensure scoring feature file does **not** require target.
- Ensure scoring feature columns match model feature columns exactly.
- If input path has no extension, pass full filename with extension to avoid ambiguity.
- if you are creating feature engineering file for training make sure that the anchor month cannot be the last month for which the data is available in the file.