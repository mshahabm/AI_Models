# Fall Risk Forecasting Model - Beta Version (QA-Validated)

## Overview

This project provides **two production-ready** fall risk forecasting models that predict next-month fall risks for healthcare plan members using machine learning. Both models use Month N features to predict Month N+1 fall occurrences with **zero data leakage** (QA-validated Feb 2026).

**Available Models:**
1. **XGBoost (Standard)** - `FallRisk_Healthplans_BetaVersion_QA_XGB.py` - Balanced performance (60% recall)
2. **RandomForest (Maximum Recall)** - `FallRisk_Healthplans_BetaVersion_QA_RF.py` - Optimized for catching falls (70-85% recall)

**Key Features**: 
- ✅ Zero data leakage (5 critical issues fixed - see `QA_FIXES_SUMMARY.md`)
- ✅ Temporal-aware feature engineering
- ✅ Automated monthly forecasting (May 2025 → Feb 2026)
- ✅ Progressive training with walk-forward validation
- ✅ CSV/Parquet support
- ✅ Risk scoring (1-10 scale)
- ✅ Comprehensive evaluation metrics

---

## Data Leakage Fixes (QA Report - Feb 2026)

**All 5 Critical Issues Fixed & Validated:**
1. ✅ Steps Aggregated Features - Now uses temporal calculation (only past data)
2. ✅ Steps Recalculation - Removed from training updates
3. ✅ Temporal-Unaware StandardScaler - Disabled for tree models
4. ✅ Hardcoded Optimal Threshold - Now calculated dynamically from training data
5. ✅ SMOTE Temporal Pooling - Disabled for temporal forecasting

**See:** `QA_FIXES_SUMMARY.md` for detailed fixes and validation

---

## Quick Start

### Installation

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn imbalanced-learn pyarrow
```

### Run the Models

**XGBoost (Standard - 60% Recall):**
```bash
python FallRisk_Healthplans_BetaVersion_QA_XGB.py
```

**RandomForest (Maximum Recall - 70-85%):**
```bash
python FallRisk_Healthplans_BetaVersion_QA_RF.py
```

Both scripts automatically process 10 months (May 2025 → February 2026), generating predictions and validation metrics for each month.

---

## Configuration

### Model Selection

**Choose based on your priorities:**

| Model | File | Recall | FPR | Best For |
|-------|------|--------|-----|----------|
| XGBoost | `_QA_XGB.py` | 60% | 3-6% | Balanced performance |
| RandomForest | `_QA_RF.py` | 70-85% | 5-8% | **Maximum fall detection** |

### Default Settings (Lines 724-743 in both files)

```python
config = {
    'algorithm': 'XGBoost',  # or 'RandomForest'
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

To predict the next month (e.g., March 2026), update **3 locations** in both files:

### 1. Lines ~206-208 - `month_order` in `prepare_forecasting_features()`
Add `'Mar_2026'` to the end of the list.

### 2. Lines ~680-683 - `month_order` in `update_training_file_with_real_data()`
Add `'Mar_2026'` to the end of the list.

### 3. Lines ~728-743 - `months_to_process` in `main()`
Add the new month entry:
```python
{'predict': 'Feb_2026', 'test_file': 'FallRisk_Test_February2026_Healthplans.CSV'},
{'predict': 'Mar_2026', 'test_file': None}  # None until data available
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
1. **Train & Predict**: Train model on all available data, generate risk scores (1-10)
2. **Test**: Compare predictions vs actual falls (when data available)
3. **Update**: Merge new data into training file for next iteration

**Forecasting Logic**: Month N features → Month N+1 predictions
- Example: April 2025 data → Predict May 2025 falls
- **Zero Leakage**: Uses only temporal features (data ≤ current month)

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

### XGBoost (Standard - 60% Recall)

**Algorithm**: XGBoost with:
- Temporal feature engineering (zero data leakage)
- Dynamic threshold calculation (F2 score optimization)
- Class weighting: `scale_pos_weight=12.0`
- Isotonic probability calibration
- SMOTE disabled to prevent temporal mixing

**Key Parameters**: `n_estimators=800`, `max_depth=3`, `learning_rate=0.01`, `threshold=0.02`

---

### RandomForest (Maximum Recall - 70-85%)

**Algorithm**: RandomForest with:
- Temporal feature engineering (zero data leakage)
- Aggressive class weighting: `{0:1, 1:150}`
- Ultra-low threshold: `0.004` (catches more falls)
- No calibration (uses raw probabilities)
- SMOTE disabled to prevent temporal mixing

**Key Parameters**: `n_estimators=800`, `max_depth=None`, `criterion='entropy'`, `class_weight={0:1, 1:150}`, `threshold=0.004`

**Trade-off**: Higher FP rate (5-8%) but catches 70-85% of falls vs 60% for XGBoost

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| File not found | Ensure training/test files exist in directory |
| Missing test file warning | Normal if data not yet available |
| Parquet support error | `pip install pyarrow` |
| SMOTE error | `pip install imbalanced-learn` |
| Memory issues | Use Parquet format or process fewer months |
| Matplotlib threading errors | Fixed - using `matplotlib.use('Agg')` backend |
| Low recall | Use RandomForest version (`_QA_RF.py`) |

---

## Performance Comparison

| Metric | XGBoost Standard | RandomForest Max Recall |
|--------|------------------|-------------------------|
| **Recall** | 60% | 70-85% ✅ |
| **False Positives** | 3-6% | 5-8% |
| **ROC-AUC** | 0.64 | 0.62-0.65 |
| **Best For** | Balanced | **Catching Falls** |

---

## Version History

**QA-Validated Version** (February 2026):
- ✅ All 5 data leakage issues fixed and validated
- ✅ Two production models: XGBoost (standard) & RandomForest (max recall)
- ✅ Temporal-aware feature engineering throughout
- ✅ Dynamic threshold calculation (F2/F3 optimization)
- ✅ Matplotlib threading fix
- Folder naming: `FallRisk_BetaVersion_*`
- Extended to February 2026

**Beta Version** (January 2026):
- Folder naming: `FallRisk_Model7_*` → `FallRisk_BetaVersion_*`
- Capitalization: `Predicting` → `predicting`
- Progressive training updates

**Period**: May 2025 → February 2026 (10 months)

---

**Last Updated**: February 3, 2026  
**QA Status**: ✅ All Issues Fixed & Validated (See `QA_FIXES_SUMMARY.md`)

