Purpose: Fall Risk Prediction Model that predicts whether a member will experience a fall in the next 1, 2, or 3 months using XGBoost classification.

Data Source

- Input: `Fall_alarm_llm_count_target_column.parquet`
- Contains 6 months of data per account (May-Oct 2025)
- Target column: Fall count (0-20) from LLM detection per month
- Removes `obs_month` column (inaccurate data)

Key Components

1. Data Loading (`load_data`)
	• Loads parquet file
	• Removes `obs_month` column
	• Performs data quality checks
	• Verifies 6 months of data per account

2. Target Creation (`create_targets`)
	• Creates binary targets for 1, 2, and 3 months ahead
	•  For each month, checks if any fall (count > 0) occurs in future months
	• Groups by account to maintain temporal order
	• Results in `target_1m`, `target_2m`, `target_3m` columns

3. Feature Engineering (`create_features`)
	• Creates lag features (1, 2, 3 months)
	• Creates rolling features (1-month window: mean and std)
	• Creates trend features (3-month slope)
	• Fills missing values appropriately
	• Total: 84 engineered features

4. Train/Test Split (`train_test_split`)
	• 50/50 split by account (not by row)
	• Ensures all rows for an account are in same set
	• Random shuffle with seed=42 for reproducibility

5. Model Training (`train_model`)
	• Trains XGBoost classifier for each horizon (1, 2, 3 months)
	• Handles class imbalance with conservative weighting
	• Optimizes prediction threshold for precision
	• Calculates metrics: accuracy, precision, recall, F1, AUC
	• Saves models with metadata

6. Visualization
	• Generates ROC-AUC curves (training and testing)
	• Generates confusion matrices (training and testing)
	•  All plots saved with timestamps

7. Report Generation (`generate_reports`)
	• Model Performance Report: All metrics for all horizons
	• Metrics Summary Report: Consolidated metrics
	• All Features Report: Detailed feature descriptions and statistics

Output Files
	• Models: `models/active_dtc_w_llm_features_{horizon}m_model.pkl`
	• ROC Curves: `results/roc_curve_{horizon}m_{dataset}_{timestamp}.png`
	• Confusion Matrices: `results/confusion_matrix_{horizon}m_{dataset}_{timestamp}.png`
	• Reports: `results/model_performance_report_{timestamp}.csv`, `results/metrics_summary_report_{timestamp}.csv`, `results/all_features_report_{timestamp}.csv`


Total Features: 12 (original) + 36 (lag) + 24 (rolling) + 12 (trend) = 84 features

Original Features (12 features) - These are the core behavioral and health indicators that can predict fall risk.
The following base features are used from the dataset:
- `avg_daily_steps`: Average daily steps for the month
- `prev_avg_daily_steps`: Previous month's average daily steps
- `steps_delta`: Change in steps from previous month
- `button_press_count`: Number of button presses in the month
- `er_dispatch_count`: Emergency response dispatch count
- `fall_alarm_count`: Fall alarm count (non-LLM)
- `sentiment_negative_count`: Count of negative sentiment operator notes
- `sentiment_positive_count`: Count of positive sentiment operator notes
- `sentiment_neutral_count`: Count of neutral sentiment operator notes
- `assist_lift_or_stand_flag_count`: Count of assistance provided events
- `fall_detect_enabled`: Binary flag if fall detection is enabled
- `age`: Member's age

Lag Features (36 features) - For each of the 12 base features, creates 3 lag features (1, 2, 3 months ago). Captures historical patterns. Recent months (1-2 months) are more predictive than older data (3 months). Helps identify trends and changes in behavior over time.

• Rolling Features (24 features) - For each of the 12 base features, creates 2 rolling features (mean and std) using 1-month window

	• Monthly rollover (1-month window): Provides immediate previous month's statistics, ensuring consistent time windows without mixing different periods
	• Rolling mean: Captures the average value over the window, smoothing out daily variations
	• Rolling std: Captures variability/volatility, which can indicate instability in health patterns

• Trend Features (12 features) - For each of the 12 base features, calculates linear trend/slope over last 3 months
	• Formula   to calculate slope of linear regression
	• Only calculated for rows with at least 3 previous months of data
	• Identifies directional trends (increasing/decreasing) in health metrics. A declining trend in steps or increasing trend in alarms may indicate worsening condition.

AI Model Library and Parameters

• Model Library: XGBoost (XGBClassifier)

XGBoost Parameters and Rationale:


| Parameter | Value |
|-----------|-------|-----|
| `n_estimators` | 150 | Number of boosting rounds. Increased from 100 to allow more learning iterations while maintaining reasonable training time. |
| `learning_rate` | 0.03 | Reduced from 0.05 to prevent overfitting. Lower learning rate with more estimators (150) provides better generalization. |
| `max_depth` | 3 | Reduced from 4  to prevent overfitting and reduce false positives. Shallower trees are more conservative and less likely to overfit to noise. |
| `subsample` | 0.8 | Row subsampling ratio. 80% of samples used per tree to add randomness and prevent overfitting. |
| `colsample_bytree` | 0.8 | Column subsampling ratio. 80% of features used per tree to add diversity and reduce overfitting. |
| `reg_alpha` | 1.0 | Increased from 0.5 - L1 regularization. Higher value penalizes large coefficients, reducing false positives and improving precision. |
| `reg_lambda` | 3.0 | Increased from 2.0 - L2 regularization. Higher value further reduces overfitting and false positives. |
| `min_child_weight` | 8 | Increased from 5 - Minimum sum of instance weight needed in a child. Higher value requires more samples for splits, making model more conservative. |
| `gamma` | 0.2 | Increased from 0.1 - Minimum loss reduction required for split. Higher value reduces overfitting by requiring more significant improvements. |
| `random_state` | 42 | Ensures reproducibility of results. |
| `eval_metric` | 'auc' | Evaluation metric - Area Under ROC Curve, appropriate for binary classification. |
| `n_jobs` | -1 | Use all available CPU cores for parallel processing. |
| `tree_method` | 'hist' | Histogram-based algorithm for faster training on large datasets. |

	• Additional Model Configuration:

	• Class Imbalance Handling:
		○ scale_pos_weight`: Dynamically calculated as `(neg_count / pos_count)  0.7`
		○  reduced by 30%: To improve precision by making the model more conservative in predicting positive class (falls), reducing false positives.

	• Threshold Optimization: Addresses feedback about low precision. Higher threshold reduces false positives.
		○ Uses precision-recall curve to find optimal threshold
		○ Prioritizes precision >= 0.3 while maximizing F1 score
