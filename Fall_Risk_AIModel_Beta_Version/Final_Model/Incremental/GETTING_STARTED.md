# Fall Risk Forecasting - Incremental Mode

**Fast, automated monthly fall risk prediction with zero data leakage**

---

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Automated Mode (Recommended)
```bash
python FallRisk_Healthplans_Incremental.py --auto
```

That's it! The script:
- Finds latest training file automatically
- Predicts next month
- Evaluates against test data (if available)
- Updates training file
- Continues to next month
- Stops when no test data available

### Run Manual Mode (Single Month)
```bash
python FallRisk_Healthplans_Incremental.py --predict_month 05_2025 --training_file FallRisk_Training_112024_To_042025_Healthplans
```

---

## Key Features

### Automation
- ✅ **Auto-detects** latest training file
- ✅ **Auto-calculates** next month to predict
- ✅ **Auto-searches** for test files
- ✅ **Auto-evaluates** predictions against real outcomes
- ✅ **Auto-updates** training file with new month
- ✅ **Auto-continues** to next month in sequence
- ✅ **Smart pause** when no test data, resumes on re-run

### Format Intelligence
- ✅ **Format-aware**: Parquet input → Parquet output, CSV input → CSV output
- ✅ **Performance**: Parquet is 5x faster and 75% smaller
- ✅ **Auto-detection**: Searches .parquet, .csv, .CSV in order
- ✅ **Conversion tool**: `python csv_to_parquet.py your_file.CSV`

### Quality Assurance
- ✅ **Stratified split**: Balanced fall representation (Audit Fix #1)
- ✅ **No patient overlap**: Complete train/validation separation
- ✅ **Early stopping**: Prevents overfitting (100 rounds)
- ✅ **Recall optimization**: Auto-finds threshold for 70% recall
- ✅ **Minimum validation**: Ensures ≥30 positive samples
- ✅ **Temporal integrity**: No future data leakage
- ✅ **All 14 QA fixes**: Fully preserved

---

## Command Reference

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--auto` | No | - | **Full automation mode** (recommended) |
| `--predict_month` | Manual only | - | Month to predict (MM_YYYY format) |
| `--training_file` | Manual only | - | Training file path |
| `--test_file` | No | Auto-detect | Test file path (optional) |
| `--algorithm` | No | XGBoost | ML algorithm |

### Examples

**Automated Mode:**
```bash
# Process all available months
python FallRisk_Healthplans_Incremental.py --auto

# With specific algorithm
python FallRisk_Healthplans_Incremental.py --auto --algorithm RandomForest
```

**Manual Mode:**
```bash
# Predict single month
python FallRisk_Healthplans_Incremental.py --predict_month 05_2025 --training_file FallRisk_Training_112024_To_042025_Healthplans

# With evaluation
python FallRisk_Healthplans_Incremental.py --predict_month 05_2025 --training_file FallRisk_Training_112024_To_042025_Healthplans --test_file FallRisk_Test_05_2025_Healthplans
```

**Help:**
```bash
python FallRisk_Healthplans_Incremental.py --help
```

---

## How Automation Works

### Automated Mode Workflow

**Single Iteration:**
1. Finds latest training file: `FallRisk_Training_*_To_*_Healthplans.*`
2. Detects format: `.parquet` or `.csv`
3. Extracts last month from filename (e.g., `042025` → `04_2025`)
4. Calculates next month (e.g., `04_2025` → `05_2025`)
5. Trains model and generates predictions
6. Searches for test file: `FallRisk_Test_05_2025_Healthplans.*`
7. **If test found**: Evaluates, updates training, goes to next month
8. **If test not found**: Saves predictions, stops, waits for data

**Multiple Iterations:**
- Continues loop until no test data available
- Maximum 20 iterations (safety limit)
- Each iteration updates training file for next
- Preserves starting month across all updates

### Example Execution

```
Current files:
- FallRisk_Training_112024_To_042025_Healthplans.parquet
- FallRisk_Test_05_2025_Healthplans.csv
- FallRisk_Test_06_2025_Healthplans.csv

Run: python FallRisk_Healthplans_Incremental.py --auto

ITERATION 1:
  Detected: Training_To_042025.parquet
  Predicting: 05_2025 (May)
  Test found: FallRisk_Test_05_2025_Healthplans.csv
  ✓ Evaluated and updated training
  → Training_To_052025.parquet created

ITERATION 2:
  Detected: Training_To_052025.parquet
  Predicting: 06_2025 (June)
  Test found: FallRisk_Test_06_2025_Healthplans.csv
  ✓ Evaluated and updated training
  → Training_To_062025.parquet created

ITERATION 3:
  Detected: Training_To_062025.parquet
  Predicting: 07_2025 (July)
  Test NOT found
  ✓ Predictions generated
  ⏸ STOPPED (waiting for July test data)

Result: Processed 2 complete months + 1 prediction
Runtime: ~30-40 minutes
```

---

## Output Files

### Format-Aware Generation

**When using Parquet input:**
```
FallRisk_Training_112024_To_052025_Healthplans.parquet  ← Updated training
FallRisk_BetaVersion_Predict_052025/
  ├── Fall_Risk_score_052025.parquet               ← Predictions (Parquet only)
  ├── ROC_AUC_Val.png
  ├── CM_Val.png, CM_Val.txt
  ├── Performance_Val.txt
  └── FeatureImportance.txt
FallRisk_BetaVersion_Test_Including_052025/        ← If test file exists
  ├── ROC_Test.png
  ├── CM_Test.png, CM_Test.txt
  └── Performance_Test.txt
```

**When using CSV input:**
Same structure but `.csv` files instead of `.parquet`

### File Contents

**Prediction Report:** `Fall_Risk_score_MMYYYY.[parquet|csv]`
- Columns: account_number, Age, demographics, Probability, Probability_Raw, Risk_Score (1-10), Risk_Category (Low/Moderate/High), Flagged, Data_Quality_Pct
- One row per patient
- Used by care teams for interventions

**Training File:** `FallRisk_Training_MMYYYY_To_MMYYYY_Healthplans.[parquet|csv]`
- Preserves starting month from input
- Includes all historical data + new month outcomes
- Ready for next month prediction

---

## File Naming Requirements

### Training Files
```
Pattern: FallRisk_Training_MMYYYY_To_MMYYYY_Healthplans.[csv|parquet]

Valid:
✓ FallRisk_Training_112024_To_042025_Healthplans.parquet
✓ FallRisk_Training_022024_To_062025_Healthplans.csv

Invalid:
✗ FallRisk_Training_To_042025_Healthplans.csv (missing start month)
✗ FallRisk_Training_11_24_To_04_25_Healthplans.csv (wrong format)
```

### Test Files
```
Pattern: FallRisk_Test_MM_YYYY_Healthplans.[csv|parquet]

Valid:
✓ FallRisk_Test_05_2025_Healthplans.csv
✓ FallRisk_Test_12_2025_Healthplans.parquet

Invalid:
✗ FallRisk_Test_5_2025_Healthplans.csv (month must be 2 digits)
✗ FallRisk_Test_052025_Healthplans.csv (missing underscore)
```

### Month Format
```
Format: MM_YYYY (with underscore)

Valid: 01_2025, 05_2025, 12_2025
Invalid: 1_2025, 05-2025, 052025, 2025_05
```

---

## QA Integrity

### All 14 QA Fixes Maintained

The script preserves all quality assurance measures:

| QA Fix | Implementation | Verification |
|--------|----------------|--------------|
| **Stratified Split** | StratifiedShuffleSplit on patient fall history | Console: "Stratification: Train=X%, Val=X%" |
| **No Overlap** | Separate patient IDs for train/validation | Console: "NO overlap" |
| **Temporal Features** | Only data ≤ current_month used | Console: "use_temporal_features=True" |
| **Early Stopping** | 100 rounds patience | Console: "Early stopping: Best iteration = XXX" |
| **Recall Optimization** | Auto-finds 70% recall threshold | Console: "Optimal threshold: X.XXXX" |
| **Min Validation** | Warns if <30 positive samples | Console: "Validation positive samples: XXX" |
| **No SMOTE** | Prevents temporal mixing | Console: "SMOTE disabled" |

**Console Markers to Watch:**
```
✓ Input format: [PARQUET|CSV]
✓ Patient split: XXX train, XXX val (NO overlap)
✓ Stratification: Train=X.XX%, Val=X.XX%
✓ use_temporal_features=True
✓ SMOTE disabled (prevents temporal mixing)
✓ Early stopping: Best iteration = XXX
✓ Optimal threshold: X.XXXX (Recall=X.XX, Precision=X.XX)
```

**If all markers present → Model is running correctly with zero data leakage!**

---

## Model Architecture

### XGBoost Configuration
- **n_estimators**: 800
- **max_depth**: 4
- **learning_rate**: 0.01
- **scale_pos_weight**: 30.0 (optimized for fall class)
- **reg_alpha**: 0.05, **reg_lambda**: 0.5
- **min_split_loss**: 0.05
- **early_stopping_rounds**: 100
- **missing**: np.nan (native NaN handling)

### Training Process
- **Patient split**: 80/20 (stratified by fall history)
- **Calibration**: DISABLED (using native XGBoost probabilities)
- **Threshold**: Auto-optimized per month (70% recall target)
- **Sample weights**: 1.0-2.0 based on data quality
- **Class weights**: 9.5x boost for fall class

---

## Performance

### Runtime
- **Single month**: ~10 min (Parquet), ~15 min (CSV)
- **With evaluation**: ~15-20 min per month
- **3-month backlog**: ~45-60 min

### Resources
- **Memory**: 2-4 GB RAM
- **Disk**: 125 MB/month (Parquet), 500 MB/month (CSV)
- **CPU**: Multi-core recommended

### File Format Comparison
| Format | I/O Speed | File Size | Recommendation |
|--------|-----------|-----------|----------------|
| **Parquet** | 5x faster | 75% smaller | ✅ Production |
| **CSV** | Baseline | Baseline | Excel viewing |

**Convert CSV to Parquet:**
```bash
python csv_to_parquet.py FallRisk_Training_112024_To_042025_Healthplans.CSV
```

---

## Use Cases

### Use Case 1: Monthly Production Run
```bash
# Every month when new test data arrives
python FallRisk_Healthplans_Incremental.py --auto

# Runtime: ~2-5 minutes
# Output: Current month predictions with evaluation
```

### Use Case 2: Catch Up on Backlog
```bash
# When you have multiple months of test data to process
python FallRisk_Healthplans_Incremental.py --auto

# Runtime: ~2-5 min per month
# Output: All available months processed sequentially
```

### Use Case 3: Generate Predictions Only
```bash
# When test data not yet available
python FallRisk_Healthplans_Incremental.py --auto

# Runtime: ~2-5 minutes
# Output: Predictions only, no evaluation
# Message: "Waiting for test file..."
```

### Use Case 4: Process Specific Month
```bash
# When you need control over specific month
python FallRisk_Healthplans_Incremental.py --predict_month 07_2025 --training_file FallRisk_Training_112024_To_062025_Healthplans

# Runtime: ~2-5 minutes
# Output: July predictions
```
---

## Troubleshooting

### Common Issues

**Training file not found**
```
Error: "No training files found"
Fix: Verify pattern: FallRisk_Training_MMYYYY_To_MMYYYY_Healthplans.[csv|parquet]
Check: dir FallRisk_Training_*_To_*_Healthplans.*
```

**Invalid month format**
```
Error: "Invalid month format"
Fix: Use MM_YYYY (e.g., 05_2025, not 5_2025)
```

**Cannot parse filename**
```
Error: "Cannot parse month from filename"
Fix: Verify exact pattern with underscores:
     FallRisk_Training_112024_To_052025_Healthplans.csv
```

**PyArrow not installed**
```
Error: "pyarrow not installed"
Fix: pip install pyarrow
```

**No test data message**
```
Message: "NO TEST DATA FOR XX_XXXX"
Action: This is normal! Wait for test data, place in directory, re-run --auto
```

---

## Best Practices

### Monthly Workflow
1. **Receive** test data: `FallRisk_Test_MM_YYYY_Healthplans.csv`
2. **Convert** to Parquet (optional but recommended): `python csv_to_parquet.py filename.CSV`
3. **Run** automation: `python FallRisk_Healthplans_Incremental.py --auto`
4. **Verify** console shows all QA markers
5. **Review** output: `Fall_Risk_score_MMYYYY.parquet`
6. **Distribute** to care teams

### Optimization Tips
- **Use Parquet format** for 5x speed boost
- **Run off-peak hours** (5 min per month)
- **Schedule monthly** (first week after test data arrival)
- **Archive outputs** for audit trail
- **Monitor QA markers** in console

### Data Management
- **Keep training files** organized by date
- **Archive predictions** monthly
- **Version control** training files
- **HIPAA compliance**: Never commit real data to git

---

## FAQ

**Q: What does automated mode do?**
```
A: Finds latest training file, predicts next month, evaluates if test data 
   available, updates training, continues to next month automatically
```

**Q: How does it know which month to predict?**
```
A: Extracts last month from training filename, adds 1 month automatically
   Example: Training_To_042025 → Predicts 05_2025
```

**Q: Can it process multiple months at once?**
```
A: Yes! If you have test files for May, June, July, automated mode 
   processes all 3 months sequentially in one run
```

**Q: What if test data isn't ready?**
```
A: Script generates predictions and stops with message:
   "Waiting for: FallRisk_Test_XX_XXXX_Healthplans"
   Re-run --auto when test data arrives - it resumes automatically
```

**Q: Does it reprocess old months?**
```
A: No! Always finds LATEST training file and predicts NEXT month only.
   Never reprocesses historical data.
```

**Q: Should I use CSV or Parquet?**
```
A: Parquet (5x faster, 75% smaller)
   Convert existing CSV: python csv_to_parquet.py your_file.CSV
```

**Q: Can I specify a custom training file?**
```
A: Yes! Use manual mode:
   python FallRisk_Healthplans_Incremental.py --predict_month 05_2025 --training_file YOUR_FILE
```

**Q: How do I verify it's working correctly?**
```
A: Check console for these markers:
   ✓ "Input format: PARQUET"
   ✓ "NO overlap"
   ✓ "Stratification: Train=X%, Val=X%"
   ✓ "Early stopping: Best iteration"
```

---

## Technical Details

### Model: XGBoost
- 800 estimators, max_depth 4
- scale_pos_weight: 30.0
- Early stopping: 100 rounds
- Calibration: DISABLED (native probabilities)
- Threshold: Auto-optimized per month (70% recall target)

### Data Processing
- Stratified 80/20 patient split (no overlap)
- Temporal features (no future data)
- NaN-aware feature engineering
- Sample weights: 1.0-2.0 by data quality
- Class weights: 9.5x minority boost

### Risk Scoring
- **Risk Score**: 1-10 (relative to threshold)
- **Risk Category**: Low (1-2), Moderate (3-6), High (7-10)
---

## Output Details

### Prediction Report Columns
- `account_number`, `account_id`, `Age`, demographics
- `Probability`: Rounded fall probability (4 decimals)
- `Probability_Raw`: Unrounded (for accurate AUC)
- `Risk_Score`: 1-10 scale
- `Risk_Category`: Low/Moderate/High
- `Flagged`: Boolean (high risk flag)
- `Data_Quality_Pct`: Data completeness percentage

### Validation Reports
- `ROC_AUC_Val.png`: ROC curve
- `CM_Val.png/txt`: Confusion matrix
- `Performance_Val.txt`: All metrics
- `FeatureImportance.txt`: Top 20 features

### Test Reports (if test file available)
- `ROC_Test.png`: Real-world ROC
- `CM_Test.png/txt`: Real-world confusion matrix
- `Performance_Test.txt`: Real-world metrics

---

## Important Notes

### HIPAA Compliance
- ⚠️ **Never commit** real healthcare data to git
- ⚠️ `.gitignore` excludes all data files
- ⚠️ Archive predictions in secure location

### Validation
- ✅ Always check console QA markers
- ✅ Review validation metrics before deployment
- ✅ Monitor recall/precision trade-off
- ✅ Track performance trends over time

### Monthly Updates
- 🔄 Run after test data arrival (usually first week of month)
- 🔄 Training file grows incrementally
- 🔄 Model improves with more data
- 🔄 Threshold auto-optimized per month

---

## Quick Reference

**Essential Command:**
```bash
python FallRisk_Healthplans_Incremental.py --auto
```

**File Requirements:**
- Training: `FallRisk_Training_MMYYYY_To_MMYYYY_Healthplans.[csv|parquet]`
- Test: `FallRisk_Test_MM_YYYY_Healthplans.[csv|parquet]` (optional)

**QA Verification:**
- ✓ "NO overlap"
- ✓ "Stratification"
- ✓ "use_temporal_features=True"
- ✓ "Input format: [PARQUET|CSV]"

**Performance:**
- ~5 min per month
- Parquet 5x faster than CSV

---

**Ready for Production!** 

For detailed technical information, see inline code comments in `FallRisk_Healthplans_Incremental.py`

