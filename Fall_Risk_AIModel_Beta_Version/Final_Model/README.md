# Fall Risk Forecasting Model - Beta Version

A machine learning system for predicting fall risk in healthcare plan members using XGBoost and historical monitoring data. This model uses **true forecasting** - predicting next month's fall risk based on current month's behavioral patterns.

## 🎯 Overview

This forecasting model predicts which members are at risk of falling in the **next month** based on their **current month's** activity patterns, device usage, and health indicators.

**Key Innovation**: Unlike concurrent prediction models, this system performs true time-series forecasting:
- **Month N features** → Predict **Month N+1 falls**
- Example: April 2025 data → Predicts May 2025 fall risk

## ✨ Features

- ✅ **True Forecasting**: Predict next month falls using current month data
- ✅ **Automated Pipeline**: 25-step automated workflow for continuous model updates
- ✅ **Multiple Algorithms**: XGBoost (default), RandomForest, GradientBoosting, LogisticRegression
- ✅ **Dual Format Support**: CSV and Parquet files (automatic detection)
- ✅ **Imbalanced Data Handling**: SMOTE oversampling + optimized class weights
- ✅ **Feature Engineering**: 50+ engineered features from raw data
- ✅ **Comprehensive Reporting**: ROC-AUC curves, confusion matrices, feature importance
- ✅ **Production Ready**: Error handling, validation, detailed logging

## 📊 What It Does

### Input
- **Training Data**: Historical monthly features (steps, alarms, assistance, sentiment, etc.)
- **Member Demographics**: Age, health plan, brand information
- **Fall Outcomes**: Historical fall counts per month

### Output
- **Risk Scores**: 1-10 scale for each member
- **Risk Categories**: High (7-10), Moderate (3-6), Low (1-2)
- **Performance Reports**: Validation & testing metrics
- **Feature Importance**: Top 20 predictive features

### Model Performance Targets
- **ROC-AUC**: Target >0.75
- **Recall**: Optimized to catch maximum falls (minimize false negatives)
- **Precision**: Balanced to reduce unnecessary interventions

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install -r requirements.txt
```

**Required Python packages**:
- pandas, numpy
- scikit-learn, xgboost
- imbalanced-learn (SMOTE)
- matplotlib, seaborn (visualization)
- pyarrow (Parquet support)

### Basic Usage

1. **Prepare your data files**:
   - Training file: `FallRisk_Training_112024_To_042025_Healthplans.CSV` (or `.parquet`)
   - Test files: `FallRisk_Test_May2025_Healthplans.CSV` (and subsequent months)

2. **Run the automated pipeline**:

```bash
python FallRisk_Healthplans_BetaVersion.py
```

3. **Check outputs**:
   - **Predictions**: `FallRisk_Model7_PredictingMay2025/FallRiskForecast_May_2025_Healthplans.CSV`
   - **Reports**: ROC curves, confusion matrices, feature importance
   - **Updated Training**: `FallRisk_Training_112024_To_052025_Healthplans.CSV`

## 📁 File Structure

### Input Files Format

**Training File Structure**:
```
account_number | account_id | age | brand | health_plan | 
avg_daily_steps_Nov_2024 | avg_daily_steps_Dec_2024 | ... |
fall_alarm_count_Nov_2024 | fall_alarm_count_Dec_2024 | ... |
fall_count_Nov_2024 | fall_count_Dec_2024 | ... |
Steps_mean | Steps_median | Steps_divergence | Steps_Max
```

**Required Columns**:
- `account_number`: Unique member identifier
- `age` or `Age`: Member age
- Monthly feature columns with pattern: `{feature_name}_{Month_Year}`
  - `avg_daily_steps_*`
  - `fall_alarm_count_*`
  - `assist_count_*`
  - `button_press_count_*`
  - `er_dispatch_count_*`
  - `sentiment_*_count_*`
  - `fall_count_*` (target variable)

**Optional Columns**:
- `account_id`, `brand`, `health_plan`
- `Steps_mean`, `Steps_median`, `Steps_divergence`, `Steps_Max`

### Output Files

**Forecast Report** (`FallRiskForecast_*.CSV`):
```
account_number | account_id | Age | brand | health_plan | 
member name | care manager | Risk_Score | Risk_Category
```

- `Risk_Score`: 1-10 scale (1=lowest, 10=highest)
- `Risk_Category`: Low, Moderate, High

## ⚙️ Configuration

### Modify Algorithm or Parameters

Edit the `CONFIG` section in `FallRisk_Healthplans_BetaVersion.py` (lines 1275-1292):

```python
config = {
    'algorithm': 'XGBoost',  # Options: 'XGBoost', 'RandomForest', 'GradientBoosting', 'LogisticRegression'
    'initial_training_file': 'FallRisk_Training_112024_To_042025_Healthplans.CSV',
    'output_base_dir': 'FallRisk_Model7',
    
    # Months to process
    'months_to_process': [
        {'predict': 'May_2025', 'test_file': 'FallRisk_Test_May2025_Healthplans.CSV'},
        {'predict': 'Jun_2025', 'test_file': 'FallRisk_Test_June2025_Healthplans.CSV'},
        # ... add more months
    ]
}
```

### Tune Model Hyperparameters

For XGBoost tuning, modify `_create_model()` method (lines 195-223):

```python
xgb.XGBClassifier(
    n_estimators=800,        # Number of trees
    max_depth=3,             # Tree depth
    learning_rate=0.01,      # Learning rate
    scale_pos_weight=12.0,   # Class imbalance handling
    # ... see code for all parameters
)
```

## 🔄 Automated Workflow

The script runs a **25-step automated pipeline**:

### For Each Month:

**Step A: Train & Predict**
1. Load training data
2. Prepare forecasting features
3. Train model with SMOTE oversampling
4. Generate validation reports
5. Predict fall risk for next month
6. Save forecast report

**Step B: Test Against Real Data**
1. Load test data with actual outcomes
2. Compare predictions vs actuals
3. Generate testing metrics
4. Save performance reports

**Step C: Update Training File**
1. Add real data to training set
2. Recalculate feature statistics
3. Save updated training file
4. Ready for next month

## 📈 Understanding Outputs

### Risk Score Interpretation

| Score | Category | Action |
|-------|----------|--------|
| 9-10 | High | Immediate intervention, care plan review |
| 7-8 | High | Enhanced monitoring, preventive measures |
| 5-6 | Moderate | Regular monitoring, education |
| 3-4 | Moderate | Standard monitoring |
| 1-2 | Low | Routine care |

### Feature Importance Reports

Three importance types are calculated:

1. **Gain**: How much each feature improves predictions (most important)
2. **Weight**: How often each feature is used in decisions
3. **Cover**: How many samples each feature affects

**Example Top Features**:
- `age`: Older members have higher fall risk
- `avg_daily_steps`: Lower activity correlates with falls
- `fall_alarm_count`: Past alarms predict future falls
- `er_dispatch_count`: Emergency history is predictive
- `assist_count`: Need for assistance indicates risk

### Performance Metrics

**ROC-AUC**: 0.75-0.85 is excellent for fall prediction
- Measures overall discriminative ability
- Higher = better separation of fall vs no-fall

**Confusion Matrix**:
- **True Positives (TP)**: Correctly predicted falls - intervention success
- **False Positives (FP)**: Predicted fall but didn't happen - unnecessary intervention
- **False Negatives (FN)**: Missed falls - critical to minimize
- **True Negatives (TN)**: Correctly predicted no fall

**Optimization Priority**: Minimize False Negatives (maximize Recall)

## 🛠️ Troubleshooting

### Common Issues

**❌ File Not Found Error**
```
ERROR: Initial training file not found
```
**Solution**: 
- Ensure training file exists in same directory as script
- Check file extension (.CSV or .parquet)
- Use absolute path if needed

**❌ Column Not Found**
```
KeyError: 'account_number'
```
**Solution**: 
- Verify your data has all required columns
- Check column name spelling and case sensitivity

**❌ Parquet Import Error**
```
ImportError: pyarrow not available
```
**Solution**: 
```bash
pip install pyarrow
```

**❌ SMOTE Warning**
```
Warning: imbalanced-learn not available
```
**Solution**: 
```bash
pip install imbalanced-learn
```

### Performance Issues

**Slow Training**:
- Reduce `n_estimators` in XGBoost config
- Use fewer historical months
- Switch to RandomForest (faster but less accurate)

**High Memory Usage**:
- Use Parquet format (more efficient than CSV)
- Process months sequentially instead of batching
- Reduce feature engineering complexity

**Poor Predictions**:
- Check data quality and completeness
- Increase training data (more historical months)
- Tune hyperparameters (see Configuration section)
- Try different algorithms

## 📊 Example Results

### Sample Execution Output

```
================================================================================
AUTOMATED FALL RISK FORECASTING - 25 STEPS
================================================================================

Configuration:
  Algorithm: XGBoost
  Initial Training: FallRisk_Training_112024_To_042025_Healthplans.CSV
  Months to Process: 9
  Output Directory: FallRisk_Model7_*

================================================================================
PROCESSING MONTH 1/9: May_2025
================================================================================

[STEP 1A] Training and Predicting May_2025...
  Training file: FallRisk_Training_112024_To_042025_Healthplans.CSV
  Training data shape: (2500, 145)
  Available months in data: ['Nov_2024', 'Dec_2024', 'Jan_2025', 'Feb_2025', 'Mar_2025', 'Apr_2025']
    Pairing: Nov_2024 features → Dec_2024 falls
    Pairing: Dec_2024 features → Jan_2025 falls
    Pairing: Jan_2025 features → Feb_2025 falls
    Pairing: Feb_2025 features → Mar_2025 falls
    Pairing: Mar_2025 features → Apr_2025 falls
  Training samples: 12500
  Features: 67
  Fall rate: 4.32%
  
  Class distribution: {0: 11960, 1: 540}
  Imbalance ratio: 22.15:1
  SMOTE applied (95% oversampling): 12500 -> 23300 samples
  New class distribution: {0: 11960, 1: 11340}
  
  Model training completed
  Optimal threshold: 0.220
  Validation ROC-AUC: 0.8245
  
  ✓ Forecast saved: FallRisk_Model7_PredictingMay2025/FallRiskForecast_May_2025_Healthplans.CSV
  ✓ Total predictions: 2500
  ✓ Risk distribution: {'Low': 1834, 'Moderate': 456, 'High': 210}

[STEP 1B] Testing May_2025 predictions against real data...
  Test file: FallRisk_Test_May2025_Healthplans.CSV
  Test data shape: (2500, 15)
  Matched accounts: 2500
  Predicted falls: 210
  Actual falls: 108
  Test ROC-AUC: 0.7892
  
[STEP 1C] Updating training file with May_2025 real data...
  ✓ Training file updated: FallRisk_Training_112024_To_052025_Healthplans.CSV
  ✓ Ready for next month

================================================================================
COMPLETED: May_2025
================================================================================
```

## 🧪 Validation & Testing

### Quality Checks

1. **Data Quality**:
   - ✅ No missing values in key features
   - ✅ Account numbers are unique
   - ✅ Date ranges are consistent

2. **Model Performance**:
   - ✅ ROC-AUC > 0.75 on validation
   - ✅ Recall > 70% (catching most falls)
   - ✅ False Positive Rate < 20%

3. **Prediction Sanity**:
   - ✅ Risk distribution is reasonable (not all high or all low)
   - ✅ High-risk members have elevated feature values
   - ✅ Manual review of sample predictions

### Testing Workflow

```bash
# 1. Run on historical data first
python FallRisk_Healthplans_BetaVersion.py

# 2. Review validation metrics
# Check: FallRisk_Model7_PredictingMay2025/ModelPerformance_Validation_Historical_May.txt

# 3. Compare with test data
# Check: FallRisk_Model7_Test_AgainstMay2025/ModelPerformance_Testing_May_2025.txt

# 4. Analyze feature importance
# Check: FallRisk_Model7_PredictingMay2025/FeatureImportance_Top20_May.txt

# 5. Tune if needed and re-run
```

## 🎓 Use Cases

1. **Proactive Care Management**: Identify high-risk members before falls occur
2. **Resource Allocation**: Target interventions to highest-risk members
3. **Care Plan Optimization**: Adjust care plans based on risk scores
4. **Quality Metrics**: Track fall prevention effectiveness
5. **Population Health**: Understand fall risk trends across health plans

## 📚 Advanced Topics

### Custom Feature Engineering

Add custom features in `_add_engineered_features()` method (lines 426-466):

```python
def _add_engineered_features(self, X):
    X = X.copy()
    
    # Your custom features
    if 'feature_A' in X.columns and 'feature_B' in X.columns:
        X['custom_ratio'] = X['feature_A'] / (X['feature_B'] + 1)
    
    return X
```

### Multi-Algorithm Ensemble

Run multiple algorithms and ensemble predictions:

```python
# Train multiple models
xgb_model = FallRiskForecastingModel(algorithm='XGBoost')
rf_model = FallRiskForecastingModel(algorithm='RandomForest')

# Average predictions
final_proba = (xgb_model.predict_proba(X) + rf_model.predict_proba(X)) / 2
```

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional algorithms (LightGBM, CatBoost)
- Real-time prediction API
- Dashboard visualization
- Explainability (SHAP values)
- External data integration (weather, medications)

## 📄 License

MIT License - See LICENSE file for details

## 📞 Support

For questions or issues:
1. Check troubleshooting section above
2. Review error messages carefully
3. Validate input data format
4. Check configuration settings

## 🔗 Related Documentation

- `GETTING_STARTED.md`: Quick start guide
- `requirements.txt`: Installation dependencies
- Script inline comments: Detailed implementation notes

## 📊 Performance Benchmarks

**Typical Performance** (based on 2500 members):
- Training time: 2-5 minutes per month
- Prediction time: < 10 seconds
- Memory usage: < 2GB RAM
- Storage: ~10MB per training file (Parquet)

## ⚠️ Important Notes

1. **Data Privacy**: Never commit real healthcare data to version control
2. **Model Validation**: Always validate predictions against real outcomes
3. **Regular Updates**: Retrain model monthly with new data
4. **Feature Drift**: Monitor feature distributions for data quality issues
5. **Threshold Tuning**: Adjust `optimal_threshold` based on intervention capacity

---

**Made with ❤️ for Healthcare Fall Prevention**

## 🌟 Star This Repository

If you find this tool helpful for fall risk prediction, please consider giving it a star! ⭐
