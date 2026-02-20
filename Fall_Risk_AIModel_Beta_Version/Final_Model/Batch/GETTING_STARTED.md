# Fall Risk Forecasting Model - Final Version

Predicts next month fall risk using current month behavioral patterns. **NaN-aware** with **recall-optimized thresholds** and **no data leakage**.

## 📌 Naming Convention

**This version uses digit-based month naming:**
- Format: `MM_YYYY` (e.g., `05_2025` for May 2025, `11_2024` for November 2024)
- Applies to: Column names, file names, configuration
- Benefits: Better sorting, international compatibility, simpler code

## 🎯 Overview

**True Forecasting**: Month N features → Month N+1 falls prediction

**Example**: April 2025 (`04_2025`) activity data → Predicts May 2025 (`05_2025`) fall risk

## ✨ Key Features

- ✅ **Recall-Optimized Thresholds**: Automatically finds threshold for 70% recall target
- ✅ **Stratified Patient Split**: Balanced fall representation in train/validation
- ✅ **Early Stopping**: Prevents overfitting (100 rounds patience)
- ✅ **NaN-Aware**: Distinguishes "no data" from "measured zero"
- ✅ **No Data Leakage**: Temporal features, stratified patient-level splits
- ✅ **Sample Weights**: Prioritizes complete data (1.0-2.0 weighting)
- ✅ **Class Weights**: 9.5x boost for minority class (falls)
- ✅ **Percentile Risk Scores**: 1-10 scale based on relative ranking
- ✅ **Dual Format**: CSV and Parquet support
- ✅ **Automated Pipeline**: Continuous monthly updates
- ✅ **Minimum Validation Check**: Ensures sufficient positive samples

## 🚀 Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Run
```bash
python FallRisk_Healthplans_Final.py
```

### Output
Generates two folders per month:
- `FallRisk_BetaVersion_Predict_MMYYYY/` - Predictions
  - `Fall_Risk_score_MMYYYY.CSV` - Main report
- `FallRisk_BetaVersion_Test_Including_MMYYYY/` - Testing results

## 📊 Model Performance

| Metric | Target | Strategy |
|--------|--------|----------|
| **Recall** | 70%+ | Recall-optimized thresholds |
| **FPR** | Minimize | While maintaining recall |
| **ROC-AUC** | >0.75 | Model quality indicator |

**Threshold Optimization**: Automatically finds threshold for 70% recall per month, then maximizes precision

## 🏗️ Architecture

### Model: XGBoost
**Parameters:**
- n_estimators: 800
- max_depth: 4 (increased from 3)
- scale_pos_weight: 30.0 (optimized)
- reg_alpha: 0.05, reg_lambda: 0.5
- min_split_loss: 0.05
- early_stopping_rounds: 100
- missing: np.nan (native NaN handling)

**Calibration:** DISABLED (using native XGBoost probabilities)

### Data Processing
1. **Temporal Steps**: Only uses data ≤ current_month
2. **Historical Data Quality**: Calculated from past months only
3. **Stratified Patient Split**: 80/20 with balanced fall representation (QA Fix #1)
4. **Minimum Validation Check**: Ensures ≥30 positive samples (QA Fix #3)
5. **Early Stopping**: Prevents overfitting with validation set
6. **SMOTE**: Disabled (prevents temporal mixing)
7. **NaN Preservation**: Kept throughout feature engineering
8. **Threshold Optimization**: Automatically finds 70% recall threshold per month

### Feature Engineering (NaN-Aware)
- Age interactions (age × steps, age × assist, etc.)
- Activity flags (low_activity, missing_activity_data)
- Data quality indicators (very_low_data_flag, has_activity_data)
- Event aggregations (total_fall_events, fall_to_assist_ratio)
- All preserve NaN (distinguishes "no data" from "0")

## 📁 Data Format

### Input
```
account_number | age | avg_daily_steps_11_2024 | fall_alarm_count_11_2024 | fall_count_11_2024 | ...
```

**Note:** All month columns now use digit format: `column_MM_YYYY` (e.g., `11_2024` for November 2024)

### Output
```
account_number | Probability | Risk_Score | Risk_Category | Flagged | Data_Quality_Pct
```

## ⚙️ Configuration

Edit `config` in `main()`:
```python
config = {
    'algorithm': 'XGBoost',
    'initial_training_file': 'FallRisk_Training_112024_To_042025_Healthplans',
    'months_to_process': [
        {'predict': '05_2025', 'test_file': 'FallRisk_Test_05_2025_Healthplans'},
        {'predict': '06_2025', 'test_file': 'FallRisk_Test_06_2025_Healthplans'},
        # Add more months...
    ]
}
```

**Note:** Months use 2-digit format (05 for May, 11 for November)

## 🔍 Key QA Fixes

1. ✅ **Stratified Split** (Lines 705-723): Balanced fall representation in train/val
2. ✅ **Minimum Validation Check** (Lines 734-742): Ensures ≥30 positive samples
3. ✅ **Early Stopping** (Lines 116, 420-427): Prevents overfitting (100 rounds)
4. ✅ **Recall-Optimized Thresholds** (Lines 315-378): Auto-finds 70% recall threshold
5. ✅ **Temporal Steps** (Lines 251-274): Only uses data up to current_month
6. ✅ **Historical Data Quality** (Lines 169-210): No future data leakage
7. ✅ **SMOTE Disabled** (Line 403): Prevents temporal mixing
8. ✅ **Steps Removed After Merge** (Lines 638-640): Recalculated temporally
9. ✅ **Sample Weights** (Lines 392-401): Data completeness based
10. ✅ **Class Weights** (Lines 385-390): 9.5x minority boost
11. ✅ **Percentile Scores**: Relative ranking
12. ✅ **NaN Handling**: Throughout (Lines 218-220, 245-316)
13. ✅ **Calibration Disabled** (Lines 431-433): Native XGBoost probabilities
14. ✅ **Raw Probability Saved** (Line 776, 859): For accurate AUC calculation

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| File not found | Check path, extension (.CSV vs .csv) |
| pyarrow missing | `pip install pyarrow` |
| Column missing | Verify `account_number`, `age` columns |
| Zero recalls | Adaptive thresholds should fix this |
| High FPR (>50%) | Increase threshold for that month |
| Slow processing | Reduce n_estimators or use Parquet |

## 📈 Risk Score Interpretation

| Score | Category | % of Pop | Action |
|-------|----------|----------|--------|
| 9-10 | High | 5-10% | Immediate intervention |
| 7-8 | High | 5-10% | Enhanced monitoring |
| 5-6 | Moderate | 15-20% | Regular monitoring |
| 3-4 | Moderate | 20-25% | Standard care |
| 1-2 | Low | 40-55% | Routine care |

## 💻 Code Stats

- **Script:** `FallRisk_Healthplans_Final.py`
- **Lines:** 903
- **Naming:** Digit-based (MM_YYYY format)
- **Functions:** 12
- **Classes:** 1
- **Dependencies:** 6 core packages
- **QA Fixes Applied:** 14 major improvements

## 📚 Documentation

- `README.md` - Full documentation
- `GETTING_STARTED.md` - This file
- `requirements.txt` - Dependencies
- `CODE_VALIDATION_REPORT.md` - QA validation
- `FINAL_NAMING_CONVENTION.md` - Folder naming spec

## 🔗 Related Files

- `FallRisk_Healthplans_Final.py` - Main script (903 lines)
- Naming convention: Digit-based months (05_2025, 11_2024, etc.)
- Audit compliance: Issue #1 (Stratification) fixed

## ⚠️ Important

1. **HIPAA Compliance**: Never commit real healthcare data
2. **Validation**: Always validate predictions before deployment
3. **Monthly Updates**: Retrain with new data monthly
4. **Monitor Performance**: Track metrics over time

---

**Ready to forecast fall risk!** 🚀

For detailed docs, see `README.md`. For troubleshooting, check inline comments in the script.
