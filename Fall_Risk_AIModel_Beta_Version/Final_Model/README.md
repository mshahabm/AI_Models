# Fall Risk Forecasting Model - Beta Version

## Overview

**FallRisk_Healthplans_BetaVersion.py** is an automated fall risk forecasting system that predicts next-month fall risks for healthcare plan members using XGBoost machine learning. It uses Month N features to predict Month N+1 fall occurrences.

**Key Features**: Automated monthly forecasting (May 2025 → Feb 2026) | Progressive training | CSV/Parquet support | Risk scoring (1-10 scale) | Comprehensive evaluation metrics

---

## Quick Start

### Installation

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn imbalanced-learn pyarrow
```

### Run the Model

```bash
python FallRisk_Healthplans_BetaVersion.py
```

The script automatically processes 10 months (May 2025 → February 2026), generating predictions and validation metrics for each month.

---

## Configuration

### Default Settings (Lines 558-574)

```python
config = {
    'algorithm': 'XGBoost',
    'initial_training_file': 'FallRisk_Training_112024_To_042025_Healthplans.CSV',
    'output_base_dir': 'FallRisk_BetaVersion',
    'months_to_process': [
        {'predict': 'May_2025', 'test_file': 'FallRisk_Test_May2025_Healthplans.CSV'},
        # ... continues through February 2026
    ]
}
```

**Output Folders**:
- Predictions: `FallRisk_BetaVersion_predicting<Month><Year>`
- Testing: `FallRisk_BetaVersion_Test_Against<Month><Year>`

---

## Extending to Future Months

To predict the next month (e.g., February 2026), update **3 locations** in the code:

### 1. Lines 145-147 - `month_order` in `prepare_forecasting_features()`
Add `'Feb_2026'` to the end of the list.

### 2. Lines 514-516 - `month_order` in `update_training_file_with_real_data()`
Add `'Feb_2026'` to the end of the list.

### 3. Lines 562-572 - `months_to_process` in `main()`
Add the new month entry:
```python
{'predict': 'Jan_2026', 'test_file': 'FallRisk_Test_January2026_Healthplans.CSV'},
{'predict': 'Feb_2026', 'test_file': None}  # None until data available
```

**Note**: Set `test_file` to `None` for future months without data yet.

---

## Output Files

### Prediction Folders: `FallRisk_BetaVersion_predicting<Month><Year>/`
- `FallRiskForecast_<Month>_<Year>_Healthplans.CSV` - Risk scores (1-10) for all members
- `ROC_AUC_Val_<Month>.png` - ROC-AUC curve
- `CM_Val_<Month>.png/txt` - Confusion matrix
- `Performance_Val_<Month>.txt` - Metrics summary
- `FeatureImportance_<Month>.txt` - Top 20 features

### Testing Folders: `FallRisk_BetaVersion_Test_Against<Month><Year>/`
- `ROC_Test_<Month>.png` - Test ROC-AUC curve
- `CM_Test_<Month>.png/txt` - Test confusion matrix
- `Performance_Test_<Month>.txt` - Test metrics

### Progressive Training Files
```
FallRisk_Training_112024_To_042025_Healthplans.CSV  (initial)
FallRisk_Training_112024_To_052025_Healthplans.CSV  (after May)
...automatically updated each month
```

---

## Workflow

**Three-Step Process Per Month**:
1. **Train & Predict**: Train XGBoost model, generate risk scores (1-10)
2. **Test**: Compare predictions vs actual falls (when data available)
3. **Update**: Merge new data into training file for next iteration

**Forecasting Logic**: Month N features → Month N+1 predictions
- Example: April 2025 data → Predict May 2025 falls

---

## Utility Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| **Extract_Monthly_Data.py** | Extract specific month from dataset | `python Extract_Monthly_Data.py` |
| **realign_rows_columns.py** | Align data structure to reference file | `python realign_rows_columns.py` |
| **Parquet_To_CSV.py** | Convert Parquet to CSV | `python Parquet_To_CSV.py` |
| **Data_Transformation_Healthplans.py** | Transform raw data for model | `python Data_Transformation_Healthplans.py` |

**realign_rows_columns.py** - Aligns January 2026 data to December 2025 structure:
- Matches row sequence by `account_number` and `health_plan`
- Keeps missing accounts with empty values
- Saves new accounts separately

---

## Risk Score Interpretation

**Scale**: 1-10 (calculated as `probability × 9 + 1`)

| Score | Category | Action |
|-------|----------|--------|
| 1-2 | Low | Standard monitoring |
| 3-6 | Moderate | Enhanced monitoring |
| 7-10 | High | Immediate intervention |

---

## Model Configuration

**Algorithm**: XGBoost with:
- Class weighting and SMOTE for imbalance
- Probability calibration
- Engineered features (age interactions, activity ratios, event totals)
- Optimal threshold: 0.22

**Key Parameters**: `n_estimators=800`, `max_depth=3`, `learning_rate=0.01`, `scale_pos_weight=12.0`

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| File not found | Ensure training/test files exist in directory |
| Missing test file warning | Normal if data not yet available |
| Parquet support error | `pip install pyarrow` |
| SMOTE error | `pip install imbalanced-learn` |
| Memory issues | Use Parquet format or process fewer months |

---

## Version History

**Beta Version** (Current):
- Folder naming: `FallRisk_Model7_*` → `FallRisk_BetaVersion_*`
- Capitalization: `Predicting` → `predicting`
- Extended to February 2026
- Progressive training updates

**Period**: May 2025 → February 2026 (10 months)

---

**Last Updated**: January 2026
