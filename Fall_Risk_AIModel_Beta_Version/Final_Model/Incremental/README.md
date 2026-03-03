# Fall Risk Forecasting - Incremental Mode

This folder contains the production incremental pipeline for monthly fall risk forecasting.

- **Script:** `FallRisk_Healthplans_Incremental.py`
- **Mode:** Incremental monthly processing with auto-loop (`--auto`)
- **Status:** Production-ready

---

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run automated mode:

```bash
python FallRisk_Healthplans_Incremental.py --auto
```

Run a single month manually:

```bash
python FallRisk_Healthplans_Incremental.py --predict_month 05_2025 --training_file FallRisk_Training_112024_To_042025_Healthplans
```

---

## What Changed

The incremental workflow and output schema now include:

1. **Script rename support**
   - Uses `FallRisk_Healthplans_Incremental.py` (not copy/manual variants).

2. **Member roster updates in incremental training generation**
   - Adds new members present in the latest real monthly file.
   - Removes canceled members not present in the latest real monthly file.

3. **Brand column normalization**
   - Uses a single `brand` identity column.
   - Legacy monthly brand columns like `brand_MM_YYYY` are collapsed into `brand` and removed.

---

## Automation Behavior

On each iteration (`--auto`):

1. Finds latest training file: `FallRisk_Training_*_To_*_Healthplans.*`
2. Extracts last trained month from filename
3. Calculates next month
4. Trains model and predicts next month risk
5. Searches test file: `FallRisk_Test_MM_YYYY_Healthplans.*`
6. If test exists:
   - Evaluates predictions
   - Builds next training file with incremental roster update (new members added, canceled removed)
   - Continues to next month
7. If test does not exist:
   - Saves prediction output
   - Stops and waits for next run

---

## Output Files

For month `MMYYYY` (example: `052025`):

- `FallRisk_BetaVersion_Predict_MMYYYY/Fall_Risk_score_MMYYYY.[parquet|csv]`
- `FallRisk_BetaVersion_Predict_MMYYYY/ROC_AUC_Val.png`
- `FallRisk_BetaVersion_Predict_MMYYYY/CM_Val.png`
- `FallRisk_BetaVersion_Predict_MMYYYY/Performance_Val.txt`
- `FallRisk_BetaVersion_Predict_MMYYYY/FeatureImportance.txt`
- `FallRisk_BetaVersion_Test_Including_MMYYYY/*` (if test file exists)

Training update output:

- `FallRisk_Training_[START]_To_[END]_Healthplans.[parquet|csv]`

---

## Prediction Report Columns

The generated score report includes:

- `account_number`
- `account_id`
- `Age`
- `brand`
- `health_plan`
- `member name`
- `care manager`
- `Data_Quality_Pct`
- `Risk_Score`
- `Risk_Category`

Internal prediction columns (`Probability`, `Probability_Raw`, `Flagged`) are used for evaluation logic and are not written in the final score report file.

---

## File Naming Rules

Training files:

```text
FallRisk_Training_MMYYYY_To_MMYYYY_Healthplans.[csv|parquet]
```

Test files:

```text
FallRisk_Test_MM_YYYY_Healthplans.[csv|parquet]
```

Month format:

```text
MM_YYYY
```

---

## Commands

Automated:

```bash
python FallRisk_Healthplans_Incremental.py --auto
```

Manual:

```bash
python FallRisk_Healthplans_Incremental.py --predict_month MM_YYYY --training_file FILE
```

Help:

```bash
python FallRisk_Healthplans_Incremental.py --help
```

---

For additional operational steps, see `GETTING_STARTED.md`.
