# Getting Started — Fall Risk Forecasting (DTC)

This guide gets you from zero to running the fall risk forecasting pipeline (`fall_risk_DTC.py`) in **incremental mode**: process one new month at a time (~10–15 min per month).

---

## 1. Prerequisites

- **Python**: 3.8+ (3.10+ recommended)
- **Required packages**: pandas, numpy, scikit-learn, xgboost
- **Optional**: pyarrow (for Parquet), imbalanced-learn (SMOTE), matplotlib, seaborn (plots)

Install core dependencies:

```bash
pip install pandas numpy scikit-learn xgboost
```

For Parquet support and optional features:

```bash
pip install pyarrow matplotlib
pip install imbalanced-learn seaborn   # optional
```

---

## 2. Input Files

The script expects **training** (and optionally **test**) data in a specific format.

### Training file

- **Naming**: `FallRisk_Training_<StartMMYYYY>_To_<EndMMYYYY>_DTC.<ext>`
  - Example: `FallRisk_Training_022024_To_042025_DTC.parquet` or `.csv`
- **Content**: One row per member (`account_number`). Columns include:
  - ID: `account_number`, optional: `brand`, `age`/`Age`, etc.
  - Feature columns per month: `<feature>_MM_YYYY` (e.g. `avg_daily_steps_11_2024`, `fall_count_12_2024`).
  - Supported months in code: `11_2024` through `02_2026`.
- **Format**: CSV or Parquet. Output format matches input (CSV in → CSV out, Parquet in → Parquet out).

### Test file (optional, for evaluation)

- **Naming**: `FallRisk_Test_<MM_YYYY>_DTC.<ext>`
  - Example: `FallRisk_Test_05_2025_DTC.parquet`
- **Content**: Same structure as training, with the **target month** columns (e.g. `fall_count_05_2025`). Used to evaluate predictions and to update the training file with the new month.

Place training and test files in the **same directory** as the script (or use full paths in the arguments).

---

## 3. Run the Script

### Manual mode (one month)

Specify the month to predict and the training file:

```bash
python fall_risk_DTC.py --predict_month 05_2025 --training_file FallRisk_Training_022024_To_042025_DTC
```

- Month format: `MM_YYYY` (e.g. `05_2025`).
- Training file: path with or without extension; script auto-detects `.csv`, `.CSV`, or `.parquet`.
- Optional: `--test_file <path>` to evaluate and update training. If omitted, the script looks for `FallRisk_Test_05_2025_DTC.<ext>` in the current directory.

Optional test file and algorithm:

```bash
python fall_risk_DTC.py --predict_month 05_2025 --training_file FallRisk_Training_022024_To_042025_DTC --test_file FallRisk_Test_05_2025_DTC --algorithm XGBoost
```

Algorithms: `XGBoost` (default), `RandomForest`, `GradientBoosting`, `LogisticRegression`.

### Automated mode

Process as many months as possible in sequence (finds latest training file, next month, and matching test file):

```bash
python fall_risk_DTC.py --auto
```

- Finds the latest `FallRisk_Training_*_To_*_DTC.*` file.
- Computes next month (e.g. after `04_2025` → `05_2025`).
- If `FallRisk_Test_<month>_DTC.<ext>` exists: trains, predicts, evaluates, then updates the training file and repeats for the next month (up to 20 iterations).
- If no test file: generates predictions only and exits; re-run with `--auto` when the test file is available.

---

## 4. Outputs

- **Predictions**: `FallRisk_BetaVersion_Predict_<MMYYYY>/`
  - `Fall_Risk_score_<MMYYYY>.parquet` or `.csv`: one row per member with `account_number`, `Probability`, `Risk_Score`, `Flagged`, `Data_Quality_Pct`, etc.
  - ROC/confusion matrix plots and text: `ROC_AUC_Val.png`, `CM_Val.png`, `Performance_Val.txt`, `FeatureImportance.txt`, etc.
- **Testing** (if test file used): `FallRisk_BetaVersion_Test_Including_<MMYYYY>/` with test ROC, CM, and performance files.
- **Updated training** (if test file used): new file like `FallRisk_Training_022024_To_052025_DTC.parquet`; use this as `--training_file` for the next month.

---

## 5. Next Steps

- Use the **updated training file** from the output as the next `--training_file` when running the following month.
- For data availability checks before modeling, use `EDA_DTC.py` (see project README).
- Adjust `--algorithm` or add new months in the script’s `month_order` if your data range changes.
