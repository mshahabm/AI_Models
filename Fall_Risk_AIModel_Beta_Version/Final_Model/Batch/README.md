# Fall Risk Forecasting Model - Final Version

**NaN-aware fall risk prediction with recall-optimized thresholds and no data leakage.**

## 📌 Naming Convention

**This version uses digit-based month naming:**
- **Format:** `MM_YYYY` (e.g., `05_2025`, `11_2024`)
- **Columns:** `avg_daily_steps_05_2025`, `fall_count_11_2024`
- **Files:** `FallRisk_Test_05_2025_Healthplans.CSV`
- **Config:** `{'predict': '05_2025', ...}`

## 🎯 What It Does

Predicts next month fall risk using current month behavioral patterns.

**Forecasting:** Month N features → Month N+1 falls prediction  
**Example:** April 2025 (`04_2025`) data → Predicts May 2025 (`05_2025`) falls

## ✨ Features

- ✅ **Recall-Optimized Thresholds**: Auto-finds threshold for 70% recall target
- ✅ **Stratified Patient Split**: Balanced fall representation in train/validation
- ✅ **Early Stopping**: Prevents overfitting (100 rounds patience)
- ✅ **NaN-Aware**: Distinguishes "no data" from "measured 0"
- ✅ **No Data Leakage**: Temporal features, stratified patient-level splits
- ✅ **Sample Weights**: Prioritizes complete data (1.0-2.0×)
- ✅ **Class Weights**: 9.5× boost for fall class
- ✅ **Percentile Scores**: Risk scores 1-10 based on ranking
- ✅ **CSV/Parquet**: Dual format support
- ✅ **Automated Pipeline**: Continuous monthly updates
- ✅ **Minimum Validation Check**: Ensures ≥30 positive samples

## 🚀 Quick Start

### Install
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn pyarrow imbalanced-learn
```

### Run
```bash
python FallRisk_Healthplans_Final.py
```

### Output Folders
- `FallRisk_BetaVersion_Predict_MMYYYY/` - Predictions
  - `Fall_Risk_score_MMYYYY.CSV` - Main report
  - ROC, confusion matrix, feature importance
- `FallRisk_BetaVersion_Test_Including_MMYYYY/` - Testing results

## 📊 Model Performance

| Metric | Target | Strategy |
|--------|--------|----------|
| **Recall** | 70%+ | Recall-optimized thresholds per month |
| **FPR** | Minimize | While maintaining recall target |
| **FNR** | <30% | By achieving 70%+ recall |
| **ROC-AUC** | >0.75 | Model quality indicator |

### Threshold Optimization Strategy

**Automatic per-month optimization:**
1. Searches 300 thresholds between 0.1th and 95th percentile
2. Finds all thresholds achieving ≥70% recall
3. Among valid thresholds, selects one with best precision
4. Falls back to maximum recall if 70% unattainable
5. Stores optimized threshold per month in `adaptive_thresholds` dict

## 🏗️ Architecture

### Model: XGBoost
- **n_estimators**: 800
- **max_depth**: 4 (increased for better pattern capture)
- **scale_pos_weight**: 30.0 (optimized for fall class)
- **reg_alpha**: 0.05, **reg_lambda**: 0.5
- **min_split_loss**: 0.05
- **early_stopping_rounds**: 100
- **missing**: np.nan (native NaN handling)

### Calibration
- **Status**: DISABLED
- **Reason**: Native XGBoost probabilities are well-calibrated
- **Benefit**: More reliable probability estimates

### Data Pipeline
1. **Temporal Steps**: Only data ≤ current_month (prevents leakage)
2. **Stratified Patient Split**: 80/20 with balanced fall representation (QA Fix #1)
3. **Minimum Validation Check**: Ensures ≥30 positive samples (QA Fix #3)
4. **Early Stopping**: Uses validation set to prevent overfitting
5. **Sample Weights**: Based on data completeness
6. **Class Weights**: 9.5× minority boost
7. **NaN Preservation**: Throughout feature engineering
8. **Threshold Optimization**: Auto-finds 70% recall threshold per month

## 📁 Data Format

### Input
```
account_number | age | avg_daily_steps_MM_YYYY | fall_alarm_count_MM_YYYY | fall_count_MM_YYYY
```

**Examples:**
- `avg_daily_steps_11_2024` (November 2024)
- `fall_count_05_2025` (May 2025)
- `assist_count_01_2026` (January 2026)

**Note:** All month columns use 2-digit month format (01-12) followed by 4-digit year

### Output
```
account_number | Probability | Risk_Score | Risk_Category | Flagged | Data_Quality_Pct
```

**Risk Categories:**
- **High (7-10)**: Immediate/enhanced intervention
- **Moderate (3-6)**: Regular/standard monitoring
- **Low (1-2)**: Routine care

## 🔍 Key Implementation Details

### QA Fixes (No Data Leakage)

1. **Stratified Patient Split** (Lines 705-723): Balanced fall representation in train/validation
2. **Minimum Validation Check** (Lines 734-742): Ensures ≥30 positive samples
3. **Early Stopping** (Lines 116, 420-427): Prevents overfitting (100 rounds)
4. **Recall-Optimized Thresholds** (Lines 315-378): Auto-finds 70% recall threshold
5. **Temporal Steps** (Lines 251-274): Only uses months ≤ current_month
6. **Historical Quality** (Lines 169-210): Past months only
7. **SMOTE Disabled**: Prevents temporal mixing
8. **Steps Recalculation**: After merge, recalculated temporally
9. **Calibration Disabled**: Native XGBoost probabilities
10. **Raw Probability Saved**: For accurate AUC calculation

### NaN Handling (Lines 218-316)

```python
# Keep NaN to distinguish "no data" from "measured 0"
X['age_x_steps'] = np.where(X['avg_daily_steps'].notna(), 
                            X['age'] * X['avg_daily_steps'], 
                            np.nan)  # NaN if no data
```

**All engineered features preserve NaN appropriately.**

### Adaptive Thresholds (Lines 101-102, 315-378)

**Automatic Optimization:**
```python
# Default threshold
self.optimal_threshold = 0.08

# Thresholds optimized per month during training
self.adaptive_thresholds = {}  # Populated automatically

# Optimization function finds threshold for 70% recall
def find_optimal_threshold(self, X, y, target_recall=0.70, predict_month=None):
    # Searches 300 thresholds
    # Finds all achieving ≥70% recall
    # Selects one with best precision
    # Stores in adaptive_thresholds[predict_month]
```

**Automatic per-month optimization replaces manual thresholds**

## 📈 Feature Importance

**Top Predictive Features:**
1. `age` - Older age = higher risk
2. `avg_daily_steps` - Lower activity = higher risk
3. `fall_alarm_count` - Past alarms predict future
4. `Steps_mean_temporal` - Activity patterns
5. `historical_data_quality` - Data completeness
6. `age_x_steps` - Age-activity interaction
7. `total_fall_events` - Cumulative event history

## ⚙️ Configuration

### Basic
```python
config = {
    'algorithm': 'XGBoost',
    'initial_training_file': 'FallRisk_Training_112024_To_042025_Healthplans',
    'months_to_process': [
        {'predict': '05_2025', 'test_file': 'FallRisk_Test_05_2025_Healthplans'},
        {'predict': '06_2025', 'test_file': 'FallRisk_Test_06_2025_Healthplans'}
        # Months use 2-digit format: 01-12
    ]
}
```

### Advanced (Lines 110-118)
```python
xgb.XGBClassifier(
    n_estimators=800, max_depth=4, learning_rate=0.01,
    scale_pos_weight=30.0, reg_alpha=0.05, reg_lambda=0.5,
    min_split_loss=0.05, early_stopping_rounds=100
)
```

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| Low recall | ✓ Automatic threshold optimization for 70% target |
| FPR too high | Model optimizes precision while maintaining recall |
| Validation unstable | ✓ Stratified split + minimum 30 positive samples check |
| Overfitting | ✓ Early stopping with 100 rounds patience |
| Missing pyarrow | `pip install pyarrow` |
| Column not found | Check `account_number`, `age` columns exist |

## 📊 Output Files

### Prediction Folder Contents:
- `Fall_Risk_score_MMYYYY.CSV` - Main forecast
- `ROC_AUC_Val.png` - ROC curve
- `CM_Val.png/txt` - Confusion matrix
- `Performance_Val.txt` - Metrics
- `FeatureImportance.txt` - Top 20 features

### Forecast Report Columns:
- `account_number`, demographics
- `Probability` - Fall risk (0-1, rounded to 4 decimals for display)
- `Probability_Raw` - Unrounded probability (for accurate AUC)
- `Risk_Score` - Score (1-10)
- `Risk_Category` - Low/Moderate/High
- `Flagged` - High risk flag (using optimized threshold)
- `Data_Quality_Pct` - Data completeness

## 💻 Code Quality

- **Lines:** 903
- **Naming Convention:** Digit-based months (MM_YYYY format)
- **Syntax:** Validated ✓
- **QA Fixes:** 14 major improvements applied ✓
- **Data Leakage:** None ✓
- **Audit Compliance:** Issue #1 (Stratification) fixed ✓

## 📚 Documentation

- `README.md` - This file
- `GETTING_STARTED.md` - Quick start guide
- `requirements.txt` - Dependencies
- `CODE_VALIDATION_REPORT.md` - QA validation
- `FINAL_NAMING_CONVENTION.md` - Output structure

## 🔗 Use Cases

1. **Proactive Care**: Identify high-risk members before falls
2. **Resource Allocation**: Target interventions efficiently
3. **Care Planning**: Adjust plans based on risk
4. **Quality Metrics**: Track prevention effectiveness

## 🤝 Contributing

Areas for enhancement:
- Additional algorithms (LightGBM, CatBoost)
- SHAP explainability
- Real-time API
- Dashboard visualization
- Multi-objective threshold optimization (e.g., F2 score)
- Cross-validation for stability assessment

## 📄 License

MIT License - See LICENSE file

## ⚠️ Important Notes

1. **HIPAA**: Never commit real healthcare data
2. **Validation**: Always validate before deployment
3. **Updates**: Retrain monthly
4. **Monitoring**: Track performance metrics

---


### Performance Targets
- Recall: >70%
- FPR: <35%
- ROC-AUC: >0.75

### Key Parameters
- Class weight multiplier: 9.5×
- Sample weight range: 1.0-2.0×
- Recall target: 70%+
- Threshold optimization: Automatic per month
- Patient split: 80/20 (stratified)
- Early stopping: 100 rounds patience
- Min validation positives: 30

---

**Made for Healthcare Fall Prevention** ❤️

For quick start, see `GETTING_STARTED.md`. For full details, review inline code comments.

---

**Version:** Final (Digit-Based Naming + Recall Optimization)  
**Lines:** 903  
**Naming:** MM_YYYY format for all months  

**Status:** Production Ready ✅
