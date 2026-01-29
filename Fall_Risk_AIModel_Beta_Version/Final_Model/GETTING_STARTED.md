# Getting Started - Fall Risk Forecasting Model (Beta Version)

## 🎉 Welcome!

You now have a production-ready **fall risk forecasting system** that predicts which members will fall next month based on their current month's activity patterns.

## ⚡ Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**What gets installed**:
- `pandas`, `numpy`: Data processing
- `scikit-learn`: Machine learning
- `xgboost`: Gradient boosting (main algorithm)
- `imbalanced-learn`: SMOTE oversampling
- `matplotlib`, `seaborn`: Visualizations
- `pyarrow`: Parquet file support

### Step 2: Prepare Your Data

**Required Files**:

1. **Training File**: `FallRisk_Training_112024_To_042025_Healthplans.CSV` (or `.parquet`)
   - Contains historical monthly features (Nov 2024 - Apr 2025)
   - Format: Wide format with month-specific columns

2. **Test Files** (optional but recommended):
   - `FallRisk_Test_May2025_Healthplans.CSV`
   - `FallRisk_Test_June2025_Healthplans.CSV`
   - etc.

**File Location**: Place files in the same directory as the script.

### Step 3: Run the Script

```bash
python FallRisk_Healthplans_BetaVersion.py
```

That's it! The script will automatically:
- Train the model on historical data
- Predict fall risk for next month(s)
- Generate reports and visualizations
- Update training file with real data

---

## 📁 Understanding Your Data Format

### Input File Structure

Your training file should have this structure:

```
| account_number | age | brand | health_plan | avg_daily_steps_Nov_2024 | avg_daily_steps_Dec_2024 | ... | fall_count_Nov_2024 | fall_count_Dec_2024 | ... |
```

**Required Columns**:
- `account_number`: Unique member ID
- `age` or `Age`: Member age
- Monthly features with pattern: `{feature}_{Month_Year}`

**Supported Features**:
- `avg_daily_steps_*`: Daily step counts
- `fall_alarm_count_*`: Fall alarm triggers
- `assist_count_*`: Assistance events
- `button_press_count_*`: Emergency button presses
- `er_dispatch_count_*`: Emergency dispatches
- `sentiment_*_count_*`: Sentiment indicators
- `fall_count_*`: Actual fall outcomes (target)

**Optional Columns**:
- `account_id`, `brand`, `health_plan`
- `Steps_mean`, `Steps_median`, `Steps_divergence`, `Steps_Max`

### Output File Structure

The forecast report will have this structure:

```
| account_number | account_id | Age | brand | health_plan | member name | care manager | Risk_Score | Risk_Category |
```

- `Risk_Score`: 1-10 scale (higher = more risk)
- `Risk_Category`: Low (1-2), Moderate (3-6), High (7-10)

---

## 🔧 Configuration

### Basic Configuration

Edit the `CONFIG` section in the script (starting at line 1275):

```python
config = {
    'algorithm': 'XGBoost',  # Algorithm choice
    'initial_training_file': 'FallRisk_Training_112024_To_042025_Healthplans.CSV',
    'output_base_dir': 'FallRisk_Model7',
    
    'months_to_process': [
        {'predict': 'May_2025', 'test_file': 'FallRisk_Test_May2025_Healthplans.CSV'},
        {'predict': 'Jun_2025', 'test_file': 'FallRisk_Test_June2025_Healthplans.CSV'},
        # Add more months as needed
    ]
}
```

### Algorithm Options

Choose from 4 algorithms:

```python
'algorithm': 'XGBoost'           # Best performance (default)
'algorithm': 'RandomForest'      # Good balance, faster
'algorithm': 'GradientBoosting'  # Good alternative
'algorithm': 'LogisticRegression' # Fastest, interpretable
```

### File Format Options

The script automatically detects and supports:
- ✅ CSV files (`.csv`, `.CSV`)
- ✅ Parquet files (`.parquet`)

**Recommendation**: Use Parquet for faster processing and smaller file sizes.

---

## 📊 Expected Output

### Console Output

```
================================================================================
AUTOMATED FALL RISK FORECASTING - 25 STEPS
================================================================================

Configuration:
  Algorithm: XGBoost
  Initial Training: FallRisk_Training_112024_To_042025_Healthplans.CSV
  Months to Process: 9

================================================================================
PROCESSING MONTH 1/9: May_2025
================================================================================

[STEP 1A] Training and Predicting May_2025...
  Training samples: 12500
  Features: 67
  Fall rate: 4.32%
  SMOTE applied: 12500 -> 23300 samples
  Validation ROC-AUC: 0.8245
  ✓ Forecast saved: FallRiskForecast_May_2025_Healthplans.CSV
  ✓ Risk distribution: {'Low': 1834, 'Moderate': 456, 'High': 210}

[STEP 1B] Testing May_2025 predictions...
  Test ROC-AUC: 0.7892
  
[STEP 1C] Updating training file...
  ✓ Ready for next month

COMPLETED: May_2025
```

### Generated Files

For each month, you'll get:

**Prediction Directory**: `FallRisk_Model7_PredictingMay2025/`
- `FallRiskForecast_May_2025_Healthplans.CSV`: Member risk scores
- `FeatureImportance_Top20_May.txt`: Most predictive features
- `ROC_AUC_Validation_Historical_May.png`: ROC curve
- `ConfusionMatrix_Validation_Historical_May.png`: Confusion matrix
- `ConfusionMatrix_Validation_Historical_May.txt`: Detailed metrics
- `ModelPerformance_Validation_Historical_May.txt`: Performance summary

**Testing Directory** (if test file provided): `FallRisk_Model7_Test_AgainstMay2025/`
- `ROC_AUC_Testing_May_2025.png`: Test ROC curve
- `ConfusionMatrix_Testing_May_2025.txt`: Test metrics
- `ModelPerformance_Testing_May_2025.txt`: Test summary

**Updated Training File**: `FallRisk_Training_112024_To_052025_Healthplans.CSV`
- Original training data + May 2025 real data

---

## 🎯 Understanding Risk Scores

### Risk Score Scale (1-10)

| Score | Category | Risk Level | Recommended Action |
|-------|----------|------------|-------------------|
| 9-10 | High | Critical | Immediate intervention, care plan review |
| 7-8 | High | Elevated | Enhanced monitoring, preventive measures |
| 5-6 | Moderate | Medium | Regular monitoring, education |
| 3-4 | Moderate | Low-Medium | Standard monitoring |
| 1-2 | Low | Minimal | Routine care |

### Using Risk Scores in Practice

**High Risk (7-10)**: ~8-12% of population
- Schedule home safety assessment
- Increase monitoring frequency
- Review medications
- Provide fall prevention education
- Consider PT/OT referral

**Moderate Risk (3-6)**: ~15-25% of population
- Standard monitoring protocols
- Encourage physical activity
- Provide educational materials
- Review periodically

**Low Risk (1-2)**: ~65-75% of population
- Routine care
- General wellness check-ins

---

## 📈 Interpreting Results

### Feature Importance

Check `FeatureImportance_Top20_*.txt` to see which features drive predictions.

**Common Top Features**:
1. `age` - Older members have higher risk
2. `avg_daily_steps` - Lower activity = higher risk
3. `fall_alarm_count` - Past alarms predict future falls
4. `assist_count` - Need for assistance indicates risk
5. `er_dispatch_count` - Emergency history is predictive

### Performance Metrics

**ROC-AUC Score**:
- 0.80-0.85: Excellent discrimination
- 0.75-0.80: Very good
- 0.70-0.75: Good
- < 0.70: Consider retraining or more data

**Confusion Matrix** (from `.txt` report):
- **True Positives**: Correctly predicted falls ✓
- **False Negatives**: Missed falls ✗ (minimize these!)
- **False Positives**: False alarms (balance with FN)
- **True Negatives**: Correctly predicted no falls ✓

### What's Good Performance?

For fall prediction:
- **Recall** > 70%: Catching most falls
- **Precision** > 15-20%: Not too many false alarms
- **ROC-AUC** > 0.75: Good discrimination

**Note**: Falls are rare events (~4-5%), so expect more false positives than true positives - this is normal!

---

## 🆘 Troubleshooting

### Common Issues

#### ❌ "File not found" error

**Problem**: 
```
ERROR: Initial training file not found
```

**Solution**:
1. Check file is in same directory as script
2. Check file name spelling (case sensitive)
3. Try absolute path: `C:\full\path\to\file.CSV`
4. Verify file extension (`.CSV` or `.parquet`)

#### ❌ "Column not found" error

**Problem**:
```
KeyError: 'account_number'
```

**Solution**:
1. Open your data file in Excel/Pandas
2. Check column names match exactly
3. Verify required columns exist:
   - `account_number`
   - `age` or `Age`
   - Monthly feature columns

#### ❌ "pyarrow not available" warning

**Problem**:
```
Warning: pyarrow not available. Parquet support disabled.
```

**Solution**:
```bash
pip install pyarrow
```

#### ❌ "imbalanced-learn not available" warning

**Problem**:
```
Warning: imbalanced-learn not available
```

**Solution**:
```bash
pip install imbalanced-learn
```

### Performance Issues

#### Slow Processing

**Solutions**:
- Reduce `n_estimators` to 400 (line 206)
- Switch to RandomForest algorithm
- Use Parquet format instead of CSV

#### High Memory Usage

**Solutions**:
- Use Parquet format (more memory efficient)
- Process fewer months at once
- Close other applications

#### Poor Predictions

**Solutions**:
1. Check data quality:
   - Are there missing values?
   - Is there enough historical data?
   - Is fall rate realistic (3-6%)?

2. Tune hyperparameters:
   - Increase `n_estimators` (line 206)
   - Adjust `max_depth` (line 207)
   - Modify `learning_rate` (line 208)

3. Get more training data:
   - Add more historical months
   - Include more members

---

## ✅ Validation Checklist

Before deploying to production:

- [ ] Training file exists and is formatted correctly
- [ ] Script runs without errors
- [ ] Output files are generated
- [ ] Risk distribution looks reasonable (not all high/low)
- [ ] ROC-AUC > 0.70 on validation
- [ ] Feature importance makes clinical sense
- [ ] Test predictions validated against real outcomes
- [ ] High-risk members manually reviewed

---

## 🎓 Next Steps

### For First-Time Users

1. ✅ Run on sample/test data first
2. ✅ Review outputs and understand metrics
3. ✅ Validate predictions manually (sample 20-30 members)
4. ✅ Tune threshold if needed
5. ✅ Deploy to production

### For Regular Users

1. ✅ Run monthly with updated data
2. ✅ Monitor performance metrics
3. ✅ Track intervention outcomes
4. ✅ Refine model based on results

### For Advanced Users

1. ✅ Experiment with different algorithms
2. ✅ Add custom engineered features
3. ✅ Tune hyperparameters for your population
4. ✅ Ensemble multiple models

---

## 🔬 Example Workflow

### Month 1: May 2025

```bash
# 1. Prepare data
# - Have training file (Nov-Apr data)
# - Have May test file (optional)

# 2. Run model
python FallRisk_Healthplans_BetaVersion.py

# 3. Review outputs
# - Check FallRisk_Model7_PredictingMay2025/
# - Review risk distribution
# - Validate sample predictions

# 4. Deploy predictions
# - Share FallRiskForecast_May_2025_Healthplans.CSV with care team
# - High-risk members get enhanced monitoring

# 5. Wait for May to end, get real outcomes

# 6. Model auto-updates training file
# - FallRisk_Training_112024_To_052025_Healthplans.CSV created
```

### Month 2: June 2025

```bash
# 1. Updated training file already includes May data
# 2. Run model again (automatically uses updated file)
# 3. Repeat process
```

---

## 💡 Pro Tips

1. **Start Small**: Test on 3-6 months of data first
2. **Validate Early**: Manually review predictions before full deployment
3. **Monitor Drift**: Check if feature distributions change over time
4. **Track Outcomes**: Document whether high-risk members actually fall
5. **Iterate**: Tune model based on real-world performance
6. **Communicate**: Share feature importance with clinical team
7. **Automate**: Schedule monthly model runs
8. **Version Control**: Track model versions and performance

---

## 📚 Additional Resources

- **Full Documentation**: See `README.md`
- **Requirements**: See `requirements.txt`
- **Code Comments**: Detailed explanations in script
- **Model Details**: Check lines 176-725 in script

---

## 🤝 Getting Help

### Self-Service

1. Read error message carefully
2. Check troubleshooting section above
3. Verify data format
4. Review configuration

### Common Questions

**Q: How much historical data do I need?**
A: Minimum 3-4 months, ideal 6+ months

**Q: Can I use different file formats?**
A: Yes! CSV and Parquet both supported

**Q: How long does it take to run?**
A: 2-5 minutes per month (2500 members)

**Q: Can I predict multiple months at once?**
A: Yes! Add all months to `months_to_process` config

**Q: What if I don't have test files?**
A: Model will still predict, just no testing step

---

## ✨ Success Criteria

You're ready for production when:

- ✅ Script runs without errors
- ✅ Risk scores are generated for all members
- ✅ Distribution looks reasonable (not 100% high or low)
- ✅ ROC-AUC > 0.70
- ✅ Manual validation confirms accuracy
- ✅ Care team understands how to use scores

---

**That's it! You're ready to forecast fall risk! 🚀**

For detailed technical documentation, see `README.md`.
