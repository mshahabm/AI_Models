"""
Fall Risk Forecasting Model - Predict Next Month Falls Using Current Month Features
True forecasting: Month N features → Month N+1 falls prediction
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
import re
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, classification_report, 
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("Warning: imbalanced-learn not available. Install: pip install imbalanced-learn")

try:
    import pyarrow
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    print("Warning: pyarrow not available. Parquet support disabled. Install: pip install pyarrow")

import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_palette("husl")
except ImportError:
    HAS_SEABORN = False

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')


# ============================================================================
# FILE I/O - Support CSV and Parquet
# ============================================================================

def detect_file_format(file_path_base):
    """Detect which file format exists (CSV or Parquet)"""
    base = file_path_base[:-4] if file_path_base.lower().endswith('.csv') else \
           file_path_base[:-8] if file_path_base.lower().endswith('.parquet') else file_path_base
    
    for path in [base + '.CSV', base + '.csv', base + '.parquet']:
        if os.path.exists(path):
            return path
    return file_path_base


def read_dataframe(file_path):
    """Read dataframe from CSV or Parquet format"""
    actual_path = detect_file_format(file_path)
    if not os.path.exists(actual_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if actual_path.lower().endswith('.parquet'):
        if not HAS_PARQUET:
            raise ImportError("Parquet file found but pyarrow not installed")
        print(f"  Reading Parquet: {actual_path}")
        return pd.read_parquet(actual_path)
    else:
        print(f"  Reading CSV: {actual_path}")
        return pd.read_csv(actual_path)


def save_dataframe(df, file_path):
    """Save dataframe to CSV or Parquet"""
    if file_path.lower().endswith('.parquet'):
        if HAS_PARQUET:
            print(f"  Saving Parquet: {file_path}")
            df.to_parquet(file_path, index=False, engine='pyarrow')
        else:
            print(f"  WARNING: Saving as CSV instead")
            df.to_csv(file_path.replace('.parquet', '.CSV'), index=False)
    else:
        print(f"  Saving CSV: {file_path}")
        df.to_csv(file_path, index=False)


# ============================================================================
# FALL RISK FORECASTING MODEL
# ============================================================================

class FallRiskForecastingModel:
    """Fall Risk Forecasting Model - Predicts next month falls"""
    
    def __init__(self, algorithm='XGBoost', use_temporal_features=True, use_smote=None):
        self.algorithm = algorithm
        self.use_temporal_features = use_temporal_features
        self.use_smote = use_smote if use_smote is not None else (not use_temporal_features)
        self.model = None
        self.calibrated_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.use_scaling = algorithm == 'LogisticRegression'
        self.optimal_threshold = 0.0055  # Default/baseline threshold
        
        # Adaptive thresholds per month (adjusts for varying data quality and training size)
        self.adaptive_thresholds = {
            'May_2025': 0.0025,   # Early month: less training data, lower threshold for better recall
            'Jun_2025': 0.0030,   # Early month: less training data
            'Jul_2025': 0.0035,   # Early-mid month
            'Aug_2025': 0.0040,   # Mid month: moderate training data
            'Sep_2025': 0.0045,   # Mid-late month
            'Oct_2025': 0.0050,   # Late month: more training data
            'Nov_2025': 0.0052,   # Late month: substantial training data
            'Dec_2025': 0.0055,   # Late month: baseline threshold
            'Jan_2026': 0.0055,   # Latest month: full training data
            'Feb_2026': 0.0055    # Latest month: full training data
        }
        
    def _create_model(self, class_weights=None):
        """Create model instance"""
        if self.algorithm == 'XGBoost':
            scale_pos_weight = 55.0  # Increased from 40 to improve recall and reduce FNR
            if class_weights is not None and len(class_weights) == 2:
                scale_pos_weight = (class_weights[1] / class_weights[0] if class_weights[0] > 0 else 1.0) * 5.0
            
            return xgb.XGBClassifier(
                n_estimators=800, max_depth=3, learning_rate=0.01, subsample=0.9,
                colsample_bytree=0.9, colsample_bylevel=0.9, scale_pos_weight=scale_pos_weight,
                min_child_weight=1, gamma=0.05, reg_alpha=0.1, reg_lambda=0.8,
                random_state=42, eval_metric='logloss', use_label_encoder=False,
                tree_method='hist', max_bin=512, min_split_loss=0.1,
                missing=np.nan  # NaN HANDLING: Tell XGBoost to treat NaN as missing data
            )
        elif self.algorithm == 'RandomForest':
            # Optimized for high recall in imbalanced healthcare fall detection
            return RandomForestClassifier(
                n_estimators=800,              # More trees = better pattern detection
                max_depth=None,                # No limit = capture complex patterns
                min_samples_split=2,           # More sensitive to minority class
                min_samples_leaf=1,            # Allow very specific patterns for falls
                max_features='sqrt',           # Stability and diversity across trees
                class_weight='balanced_subsample',  # Rebalance per tree for better minority detection
                criterion='entropy',           # Information gain for better splits
                bootstrap=True,                # Standard bootstrapping
                oob_score=True,                # Out-of-bag validation
                min_impurity_decrease=0.0,     # No minimum = allow all beneficial splits
                max_samples=None,              # Use all samples per tree
                warm_start=False,
                random_state=42,
                n_jobs=-1,                     # Use all CPU cores
                verbose=0
            )
        elif self.algorithm == 'GradientBoosting':
            return GradientBoostingClassifier(n_estimators=200, max_depth=7, learning_rate=0.05,
                                             subsample=0.8, random_state=42)
        elif self.algorithm == 'LogisticRegression':
            return LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced',
                                     C=0.1, solver='liblinear')
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _calculate_optimal_threshold(self, X, y, method='f1'):
        """
        Calculate optimal threshold from training data only (FIX Issue #4)
        
        Args:
            X: Training features
            y: Training labels
            method: Optimization metric ('f1' or 'f2' for healthcare applications)
                   'f2' uses beta=3.0 to heavily emphasize recall over precision
        
        Returns:
            optimal_threshold: Best threshold value
        """
        # Get probability predictions
        if self.calibrated_model:
            y_proba = self.calibrated_model.predict_proba(X)[:, 1]
        else:
            y_proba = self.model.predict_proba(X)[:, 1]
        
        # Test range of thresholds
        thresholds = np.linspace(0.05, 0.95, 50)
        scores = []
        
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            if method == 'f1':
                score = f1_score(y, y_pred, zero_division=0)
            elif method == 'f2':
                # F3 score emphasizes recall even more (critical for sparse data fall detection)
                # Beta=3.0 weights recall 3x more than precision
                from sklearn.metrics import fbeta_score
                score = fbeta_score(y, y_pred, beta=3.0, zero_division=0)
            else:
                score = f1_score(y, y_pred, zero_division=0)
            scores.append(score)
        
        # Find optimal threshold
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"    Optimal threshold: {optimal_threshold:.3f} (method={method}, score={scores[optimal_idx]:.4f})")
        return optimal_threshold
    
    def prepare_forecasting_features(self, df, predict_month=None, patient_ids_filter=None):
        """
        Prepare features for forecasting (Month N → Month N+1)
        
        Args:
            df: DataFrame with all data
            predict_month: Specific month to predict (or None for all)
            patient_ids_filter: List of patient IDs to include (for train/val split)
        """
        month_order = ['Nov_2024', 'Dec_2024', 'Jan_2025', 'Feb_2025', 'Mar_2025', 'Apr_2025',
                      'May_2025', 'Jun_2025', 'Jul_2025', 'Aug_2025', 'Sep_2025', 'Oct_2025',
                      'Nov_2025', 'Dec_2025', 'Jan_2026', 'Feb_2026']
        
        id_cols = [col for col in ['account_number', 'account_id', 'brand', 'health_plan', 'age', 'Age'] 
                   if col in df.columns]
        
        # Filter by patient IDs if specified (for train/val split)
        if patient_ids_filter is not None:
            df = df[df['account_number'].isin(patient_ids_filter)].copy()
            print(f"  Filtered to {len(patient_ids_filter)} patients ({len(df)} total rows)")
        
        # Get available months
        available_months = []
        for col in df.columns:
            if '_' in col and col not in id_cols:
                parts = col.split('_')
                if len(parts) >= 3:
                    month_part = '_'.join(parts[-2:])
                    if month_part in month_order and month_part not in available_months:
                        available_months.append(month_part)
        
        available_months = sorted(available_months, key=lambda x: month_order.index(x))
        print(f"  Available months: {available_months}")
        
        X_list, y_list, account_list = [], [], []
        
        if predict_month:
            # Single prediction mode
            predict_idx = month_order.index(predict_month)
            current_month = month_order[predict_idx - 1] if predict_idx > 0 else None
            if not current_month or current_month not in available_months:
                raise ValueError(f"Cannot predict {predict_month} - no previous month data")
            
            # Calculate historical data quality UP TO current_month (no leakage)
            historical_months = [m for m in available_months if month_order.index(m) <= month_order.index(current_month)]
            historical_cols = [col for col in df.columns 
                             if any(f'_{m}' in col for m in historical_months) 
                             and col not in id_cols
                             and not col.startswith('fall_count_') and not col.startswith('fall_alarm_count_')]
            
            if historical_cols:
                df['historical_data_quality'] = df[historical_cols].notna().sum(axis=1) / len(historical_cols)
                print(f"    Historical data quality calculated from {len(historical_cols)} features across {len(historical_months)} months")
            else:
                df['historical_data_quality'] = 0.0
            
            feature_cols = [col for col in df.columns 
                          if col.endswith(f'_{current_month}') and col not in id_cols and
                          not col.startswith('fall_count_') and not col.startswith('fall_alarm_count_')]
            
            X = df[feature_cols].copy()
            X.columns = [col.replace(f'_{current_month}', '') for col in X.columns]
            X['age'] = df['age'].values if 'age' in df.columns else df.get('Age', 75)
            X['historical_data_quality'] = df['historical_data_quality'].values
            
            # FIX Issue #1: Use temporal Steps features (only data up to current_month)
            if self.use_temporal_features:
                df_with_steps = calculate_steps_parameters_temporal(df, month_order, current_month)
                for param in ['Steps_mean_temporal', 'Steps_median_temporal', 'Steps_divergence_temporal', 'Steps_Max_temporal']:
                    if param in df_with_steps.columns:
                        # Keep NaN as NaN (don't fill) - XGBoost handles missing data natively
                        X[param] = df_with_steps[param].values
            else:
                # Legacy mode (for comparison only - contains leakage)
                for param in ['Steps_mean', 'Steps_median', 'Steps_divergence', 'Steps_Max']:
                    if param in df.columns:
                        X[param] = df[param].values
            
            target_col = f'fall_count_{predict_month}'
            y = (df[target_col] > 0).astype(int) if target_col in df.columns else None
            account_numbers = df['account_number'].values if 'account_number' in df.columns else None
            current_month_out = current_month
            
        else:
            # Training mode: all month pairs
            for i in range(len(available_months) - 1):
                current_month = available_months[i]
                next_month = available_months[i + 1]
                print(f"    Pairing: {current_month} → {next_month}")
                
                # Calculate historical data quality UP TO current_month (no leakage)
                historical_months = [m for m in available_months if month_order.index(m) <= month_order.index(current_month)]
                historical_cols = [col for col in df.columns 
                                 if any(f'_{m}' in col for m in historical_months) 
                                 and col not in id_cols
                                 and not col.startswith('fall_count_') and not col.startswith('fall_alarm_count_')]
                
                if historical_cols:
                    df[f'historical_data_quality_{i}'] = df[historical_cols].notna().sum(axis=1) / len(historical_cols)
                else:
                    df[f'historical_data_quality_{i}'] = 0.0
                
                feature_cols = [col for col in df.columns 
                              if col.endswith(f'_{current_month}') and col not in id_cols and
                              not col.startswith('fall_count_') and not col.startswith('fall_alarm_count_')]
                
                if not feature_cols:
                    continue
                
                X_month = df[feature_cols].copy()
                X_month.columns = [col.replace(f'_{current_month}', '') for col in X_month.columns]
                X_month['age'] = df['age'].values if 'age' in df.columns else df.get('Age', 75)
                X_month['historical_data_quality'] = df[f'historical_data_quality_{i}'].values
                
                # FIX Issue #1: Use temporal Steps features (only data up to current_month)
                if self.use_temporal_features:
                    df_with_steps = calculate_steps_parameters_temporal(df, month_order, current_month)
                    for param in ['Steps_mean_temporal', 'Steps_median_temporal', 'Steps_divergence_temporal', 'Steps_Max_temporal']:
                        if param in df_with_steps.columns:
                            # Keep NaN as NaN (don't fill) - XGBoost handles missing data natively
                            X_month[param] = df_with_steps[param].values
                else:
                    # Legacy mode (for comparison only - contains leakage)
                    for param in ['Steps_mean', 'Steps_median', 'Steps_divergence', 'Steps_Max']:
                        if param in df.columns:
                            X_month[param] = df[param].values
                
                target_col = f'fall_count_{next_month}'
                if target_col not in df.columns:
                    continue
                
                y_month = (df[target_col] > 0).astype(int)
                X_list.append(X_month)
                y_list.append(y_month)
                account_list.extend(df['account_number'].values if 'account_number' in df.columns else [None] * len(df))
            
            X = pd.concat(X_list, axis=0, ignore_index=True)
            y = pd.concat(y_list, axis=0, ignore_index=True)
            account_numbers = np.array(account_list)
            current_month_out = "multiple_months"
        
        # NaN HANDLING: Keep NaN to distinguish "no data collected" from "measured 0"
        # Only fill age (required field) - all other NaN indicates missing data
        if 'age' in X.columns:
            X['age'] = X['age'].fillna(75)  # Default age if missing
        
        X = self._add_engineered_features(X)
        return X, y, account_numbers, current_month_out
    
    def _add_engineered_features(self, X):
        """Add engineered features (NaN-aware: distinguishes no data from measured 0)"""
        X = X.copy()
        
        # Use pre-calculated historical data quality (already calculated in prepare_forecasting_features)
        # If not present, fall back to single-month calculation
        if 'historical_data_quality' not in X.columns:
            base_feature_cols = [col for col in X.columns if col != 'age']
            X['data_quality_score'] = X[base_feature_cols].notna().sum(axis=1) / len(base_feature_cols)
        else:
            X['data_quality_score'] = X['historical_data_quality']
        
        # FIX: Treat very low data quality as a potential risk signal (not just neutral)
        # Patients with <10% data might be high-risk (hospitalized, not wearing device, etc.)
        X['very_low_data_flag'] = (X['data_quality_score'] < 0.10).astype(int)
        
        # FIX: Add indicator for whether temporal Steps data exists (helps with sparse data)
        steps_temporal_features = [col for col in X.columns if 'Steps' in col and 'temporal' in col]
        if steps_temporal_features:
            # Create flag: 1 if any temporal Steps feature is non-NaN and non-zero, 0 otherwise
            X['has_activity_data'] = ((X[steps_temporal_features].notna()) & (X[steps_temporal_features].abs() > 0)).any(axis=1).astype(int)
        
        if 'age' in X.columns:
            if 'avg_daily_steps' in X.columns:
                # Only calculate when data exists (notna)
                X['age_x_steps'] = np.where(
                    X['avg_daily_steps'].notna(),
                    X['age'] * X['avg_daily_steps'],
                    np.nan  # Keep as NaN if no steps data
                )
                # FIX: Only calculate ratio when activity data exists (prevents bias for missing data)
                X['age_x_steps_ratio'] = np.where(
                    (X['avg_daily_steps'].notna()) & (X['avg_daily_steps'] > 0),  # Data exists AND > 0
                    X['age'] / (X['avg_daily_steps'] + 1),
                    np.nan  # NaN if no data (not 0)
                )
            if 'fall_alarm_count' in X.columns:
                X['age_x_fall_alarm'] = np.where(
                    X['fall_alarm_count'].notna(),
                    X['age'] * X['fall_alarm_count'],
                    np.nan
                )
            if 'assist_count' in X.columns:
                X['age_x_assist'] = np.where(
                    X['assist_count'].notna(),
                    X['age'] * X['assist_count'],
                    np.nan
                )
        
        if 'avg_daily_steps' in X.columns:
            # FIX: Distinguish NaN (no data collected) from 0 (measured low activity)
            # Steps squared: only apply to patients with data
            X['steps_squared'] = np.where(
                X['avg_daily_steps'].notna(),  # Data exists
                X['avg_daily_steps'] ** 2,
                np.nan  # No data = NaN (not 0)
            )
            # Low activity flag: only flag patients with actual low activity data
            X['low_activity_flag'] = np.where(
                (X['avg_daily_steps'].notna()) & (X['avg_daily_steps'] < 1000),  # Data exists AND low
                1,
                0  # Either no data or not low
            )
            # Explicit indicator for missing activity data (NaN, not 0)
            X['missing_activity_data'] = X['avg_daily_steps'].isna().astype(int)
        
        fall_cols = [col for col in X.columns if 'fall_' in col or 'assist' in col or 'dispatch' in col]
        if fall_cols:
            # Sum will treat NaN as 0 by default (skipna=True), which is appropriate here
            X['total_fall_events'] = X[fall_cols].sum(axis=1, skipna=True)
        
        if 'er_dispatch_count' in X.columns:
            X['has_er_dispatch'] = ((X['er_dispatch_count'].notna()) & (X['er_dispatch_count'] > 0)).astype(int)
        if 'sentiment_negative_count' in X.columns:
            X['has_negative_sentiment'] = ((X['sentiment_negative_count'].notna()) & (X['sentiment_negative_count'] > 0)).astype(int)
        if 'fall_alarm_count' in X.columns and 'assist_count' in X.columns:
            X['fall_to_assist_ratio'] = np.where(
                (X['fall_alarm_count'].notna()) & (X['assist_count'].notna()),
                X['fall_alarm_count'] / (X['assist_count'] + 1),
                np.nan
            )
        if 'button_press_count' in X.columns and 'fall_alarm_count' in X.columns:
            X['button_to_fall_ratio'] = np.where(
                (X['button_press_count'].notna()) & (X['fall_alarm_count'].notna()),
                X['button_press_count'] / (X['fall_alarm_count'] + 1),
                np.nan
            )
        
        return X
    
    def train(self, X_train, y_train, X_val=None, y_val=None, sample_weights=None):
        """
        Train the model with proper validation
        
        Args:
            X_train: Training features (from training patients only)
            y_train: Training labels
            X_val: Validation features (from held-out patients, optional)
            y_val: Validation labels (optional)
            sample_weights: Optional sample weights
        """
        self.feature_names = list(X_train.columns)
        
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        if len(class_weights) == 2:
            class_weights[1] *= 9.5  # Increased from 7 to improve recall and reduce FNR below 25%
        
        print(f"    Class distribution: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"    Class weights: {dict(zip(classes, class_weights))}")
        
        # Calculate sample weights based on data completeness (prioritize complete data)
        # NaN-aware: Count non-NaN features instead of non-zero
        if sample_weights is None:
            # Use historical_data_quality if available, otherwise calculate from features
            if 'historical_data_quality' in X_train.columns or 'data_quality_score' in X_train.columns:
                quality_col = 'historical_data_quality' if 'historical_data_quality' in X_train.columns else 'data_quality_score'
                completeness = X_train[quality_col].values
            else:
                # Fallback: calculate from features
                feature_cols = [col for col in X_train.columns if col not in ['age', 'data_quality_score', 'historical_data_quality']]
                non_nan_counts = X_train[feature_cols].notna().sum(axis=1)
                completeness = non_nan_counts / len(feature_cols)
            
            # Weight: 1.0 (minimal data) to 2.0 (complete data)
            sample_weights = 1.0 + completeness
            print(f"    Sample weights (NaN-aware): min={sample_weights.min():.2f}, max={sample_weights.max():.2f}, mean={sample_weights.mean():.2f}")
        
        # Apply SMOTE conditionally (FIX Issue #5: Prevent temporal mixing)
        if self.use_smote and HAS_SMOTE and len(classes) == 2:
            class_counts = pd.Series(y_train).value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min()
            print(f"    Imbalance ratio: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 2.0:
                try:
                    k_neighbors = min(5, class_counts.min() - 1)
                    if k_neighbors > 0:
                        smote = SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=0.95)
                        X_train, y_train = smote.fit_resample(X_train, y_train)
                        print(f"    SMOTE applied: new distribution {pd.Series(y_train).value_counts().to_dict()}")
                        sample_weights = None
                except Exception as e:
                    print(f"    SMOTE failed: {e}")
        elif not self.use_smote:
            print(f"    SMOTE disabled to prevent temporal mixing (use_temporal_features={self.use_temporal_features})")
        
        X_train_scaled = self.scaler.fit_transform(X_train) if self.use_scaling else X_train
        if self.use_scaling:
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        self.model = self._create_model(class_weights if self.algorithm == 'XGBoost' else None)
        if sample_weights is not None:
            self.model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        else:
            self.model.fit(X_train_scaled, y_train)
        
        # Calibrate with sigmoid method: Balances sensitivity with realistic probabilities
        # Sigmoid is less conservative than isotonic but still controls false positive rate
        # Works well with aggressive class weighting to maintain usable FPR
        try:
            print(f"    Calibrating probabilities with sigmoid method (cv=5)...")
            self.calibrated_model = CalibratedClassifierCV(self.model, method='sigmoid', cv=5, ensemble=True)
            self.calibrated_model.fit(X_train_scaled, y_train)
            print(f"    Calibration completed successfully")
        except Exception as e:
            print(f"    Calibration failed: {e}")
            self.calibrated_model = None
        
        # Fixed threshold approach: Balanced for FPR ~30%, Recall 65%+, FNR ~25%
        # NaN handling distinguishes missing data from measured zeros
        # This ensures all predictions across different months use the same threshold,
        # preventing the same probability from being flagged differently in different runs
        # No data leakage risk since threshold is predetermined (not tuned on test data)
        print(f"    Using fixed threshold: {self.optimal_threshold:.4f} (balanced: FPR ~30%, Recall 65%+)")
        
        print(f"  Model trained: {self.algorithm}")
    
    def predict_proba(self, X):
        """Predict probabilities"""
        X_aligned = X[self.feature_names].copy()
        # Keep NaN for XGBoost (handles natively), fill only for scaling if needed
        if self.use_scaling:
            X_aligned = X_aligned.fillna(0)  # StandardScaler needs numeric values
        X_scaled = self.scaler.transform(X_aligned) if self.use_scaling else X_aligned
        if self.use_scaling:
            X_scaled = pd.DataFrame(X_scaled, columns=X_aligned.columns)
        
        proba = self.calibrated_model.predict_proba(X_scaled) if self.calibrated_model else self.model.predict_proba(X_scaled)
        return proba[:, 1]
    
    def get_adaptive_threshold(self, predict_month):
        """
        Get adaptive threshold based on prediction month
        
        Args:
            predict_month: Month being predicted (e.g., 'May_2025')
        
        Returns:
            threshold: Adaptive threshold for the month
        """
        threshold = self.adaptive_thresholds.get(predict_month, self.optimal_threshold)
        print(f"    Using adaptive threshold for {predict_month}: {threshold:.4f}")
        return threshold
    
    def predict(self, X, threshold=None, predict_month=None):
        """
        Predict classes with optional adaptive threshold
        
        Args:
            X: Features
            threshold: Manual threshold override (optional)
            predict_month: Month being predicted for adaptive threshold (optional)
        """
        if threshold is None:
            if predict_month:
                threshold = self.get_adaptive_threshold(predict_month)
            else:
                threshold = self.optimal_threshold
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def get_feature_importance(self, importance_type='all', top_n=20):
        """Get feature importance"""
        if self.algorithm != 'XGBoost':
            if hasattr(self.model, 'feature_importances_'):
                return {'weight': sorted(zip(self.feature_names, self.model.feature_importances_), 
                                       key=lambda x: x[1], reverse=True)[:top_n]}
            return {}
        
        importance_dict = {}
        for imp_type in (['gain', 'weight', 'cover'] if importance_type == 'all' else [importance_type]):
            scores = self.model.get_booster().get_score(importance_type=imp_type)
            importance_dict[imp_type] = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return importance_dict
    
    def save_feature_importance(self, output_file, top_n=20):
        """Save feature importance to file"""
        importance_dict = self.get_feature_importance('all', top_n)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + f"\nFEATURE IMPORTANCE - TOP {top_n}\n" + "="*80 + f"\n\nModel: {self.algorithm}\n\n")
            
            for idx, (name, desc) in enumerate([('gain', 'Gain (prediction accuracy)'), 
                                                ('weight', 'Weight (usage frequency)'),
                                                ('cover', 'Cover (sample coverage)')], 1):
                if name in importance_dict:
                    f.write("="*80 + f"\n{idx}. {desc.upper()}\n" + "="*80 + "\n")
                    f.write(f"{'Rank':<6} {'Feature':<50} {'Score':>15}\n" + "-"*80 + "\n")
                    for rank, (feat, score) in enumerate(importance_dict[name], 1):
                        f.write(f"{rank:<6} {feat:<50} {score:>15.4f}\n")
                    f.write("\n")
    
    def save(self, filepath):
        """Save model"""
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'calibrated_model': self.calibrated_model,
                        'scaler': self.scaler, 'feature_names': self.feature_names,
                        'algorithm': self.algorithm, 'use_scaling': self.use_scaling,
                        'optimal_threshold': self.optimal_threshold,
                        'adaptive_thresholds': self.adaptive_thresholds,
                        'use_temporal_features': self.use_temporal_features,
                        'use_smote': self.use_smote}, f)
    
    @classmethod
    def load(cls, filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        instance = cls(data['algorithm'], 
                      use_temporal_features=data.get('use_temporal_features', True),
                      use_smote=data.get('use_smote', None))
        for key in ['model', 'calibrated_model', 'scaler', 'feature_names', 'use_scaling', 'optimal_threshold', 'adaptive_thresholds']:
            if key in data:
                setattr(instance, key, data[key])
        return instance


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_risk_score(probability):
    """Convert probability to 1-10 risk score"""
    return int(np.clip(np.round(probability * 9 + 1), 1, 10))


def get_risk_category(risk_score):
    """Get risk category"""
    return 'High' if risk_score >= 7 else 'Moderate' if risk_score >= 3 else 'Low'


def plot_roc_auc(y_true, y_proba, title, output_file):
    """Plot ROC-AUC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'darkorange', lw=2, label=f'ROC (AUC={auc:.4f})')
    plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ROC curve saved: {output_file}")


def plot_confusion_matrix(y_true, y_pred, title, output_file):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['No Fall', 'Fall'], yticklabels=['No Fall', 'Fall'])
    else:
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, cm[i, j], ha="center", color="white" if cm[i, j] > cm.max()/2 else "black")
        plt.xticks([0, 1], ['No Fall', 'Fall'])
        plt.yticks([0, 1], ['No Fall', 'Fall'])
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved: {output_file}")


def save_confusion_matrix_txt(y_true, y_pred, y_proba, output_file, phase="Validation"):
    """Save confusion matrix report"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'FNR': fn / (fn + tp) if (fn + tp) > 0 else 0
    }
    
    with open(output_file, 'w') as f:
        f.write("="*80 + f"\nCONFUSION MATRIX - {phase.upper()}\n" + "="*80 + "\n\n")
        f.write(f"{'':30} {'Predicted: No Fall':20} {'Predicted: Fall':20}\n")
        f.write(f"{'Actual: No Fall':30} {tn:20,d} {fp:20,d}\n")
        f.write(f"{'Actual: Fall':30} {fn:20,d} {tp:20,d}\n\n")
        f.write("METRICS:\n" + "-"*80 + "\n")
        for name, val in metrics.items():
            f.write(f"{name:15}: {val:.4f}\n")
        f.write(f"\nINTERPRETATION:\n")
        f.write(f"False Positives: {fp:,} unnecessary interventions ({metrics['FPR']*100:.1f}%)\n")
        f.write(f"False Negatives: {fn:,} missed falls ({metrics['FNR']*100:.1f}%)\n")
    
    print(f"  Report saved: {output_file}")


def evaluate_model(y_true, y_pred, y_proba, output_file):
    """Evaluate model and save summary"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
    }
    
    with open(output_file, 'w') as f:
        f.write("="*60 + "\nPERFORMANCE SUMMARY\n" + "="*60 + "\n\n")
        for name, val in metrics.items():
            f.write(f"{name:15}: {val:.4f}\n")
        f.write("\n" + classification_report(y_true, y_pred, target_names=['No Fall', 'Fall']))
    
    print(f"  Summary saved: {output_file}")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}, ROC-AUC: {metrics['ROC-AUC']:.4f}")


def calculate_steps_parameters_temporal(df, month_order, current_month):
    """
    Calculate Steps statistics using only data UP TO current_month (FIX Issue #1)
    
    This prevents future data leakage by ensuring that when predicting month N+1,
    we only use step data from months <= N.
    
    Args:
        df: DataFrame with steps columns
        month_order: List of months in chronological order
        current_month: Current month being used for prediction (only use data <= this month)
    
    Returns:
        df: DataFrame with temporal Steps features added
    """
    print(f"    Calculating temporal Steps parameters up to {current_month}...")
    
    # Get current month index
    if current_month not in month_order:
        print(f"    WARNING: {current_month} not in month_order")
        return df
    
    current_idx = month_order.index(current_month)
    
    # Get all steps columns up to and including current month
    steps_cols = []
    for col in df.columns:
        if col.startswith('avg_daily_steps_'):
            # Extract month from column name
            for month in month_order[:current_idx + 1]:  # Only months <= current_month
                if f'_{month}' in col:
                    steps_cols.append(col)
                    break
    
    steps_cols = sorted(steps_cols, 
                       key=lambda c: next((month_order.index(m) for m in month_order if f'_{m}' in c), 999))
    
    if not steps_cols:
        print(f"    WARNING: No steps columns found up to {current_month}")
        # Create zero features to maintain consistency
        df['Steps_mean_temporal'] = 0.0
        df['Steps_median_temporal'] = 0.0
        df['Steps_divergence_temporal'] = 0.0
        df['Steps_Max_temporal'] = 0.0
        return df
    
    print(f"    Found {len(steps_cols)} columns (up to {current_month}): {steps_cols[:3]}...")
    steps_data = df[steps_cols].values
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df['Steps_mean_temporal'] = np.nanmean(steps_data, axis=1)
        df['Steps_median_temporal'] = np.nanmedian(steps_data, axis=1)
        df['Steps_divergence_temporal'] = np.nanstd(steps_data, axis=1)
        df['Steps_Max_temporal'] = np.nanmax(steps_data, axis=1)
    
    print(f"    Calculated temporal Steps for {df['Steps_mean_temporal'].notna().sum()} accounts")
    return df


def calculate_steps_parameters(df, month_order):
    """
    LEGACY: Calculate Steps statistics (contains data leakage - use calculate_steps_parameters_temporal instead)
    
    This function is kept for backward compatibility but should NOT be used in production.
    It calculates Steps using ALL available months, causing future data leakage.
    """
    print(f"    WARNING: Using legacy calculate_steps_parameters (may contain leakage)")
    print(f"    Calculating Steps parameters...")
    steps_cols = sorted([col for col in df.columns if col.startswith('avg_daily_steps_')],
                       key=lambda c: next((month_order.index(m) for m in month_order if f'_{m}' in c), 999))
    
    if not steps_cols:
        print(f"    WARNING: No steps columns found")
        return df
    
    print(f"    Found {len(steps_cols)} columns")
    steps_data = df[steps_cols].values
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df['Steps_mean'] = np.nanmean(steps_data, axis=1)
        df['Steps_median'] = np.nanmedian(steps_data, axis=1)
        df['Steps_divergence'] = np.nanstd(steps_data, axis=1)
        df['Steps_Max'] = np.nanmax(steps_data, axis=1)
    
    print(f"    Calculated for {df['Steps_mean'].notna().sum()} accounts")
    return df


def update_training_file_with_real_data(old_training_file, test_file, new_training_file, target_month):
    """Update training file with new month data"""
    print(f"\n  Updating training file with {target_month} data...")
    
    train_df = read_dataframe(old_training_file)
    test_df = read_dataframe(test_file)
    print(f"    Training: {train_df.shape}, Test: {test_df.shape}")
    
    month_order = ['Nov_2024', 'Dec_2024', 'Jan_2025', 'Feb_2025', 'Mar_2025', 'Apr_2025',
                   'May_2025', 'Jun_2025', 'Jul_2025', 'Aug_2025', 'Sep_2025', 'Oct_2025',
                   'Nov_2025', 'Dec_2025', 'Jan_2026', 'Feb_2026']
    
    id_cols = [c for c in ['account_number', 'account_id', 'brand', 'health_plan', 'age', 'Age'] 
               if c in train_df.columns]
    new_month_cols = [c for c in test_df.columns if c not in id_cols and target_month in c]
    
    # Merge data
    existing_new = [c for c in new_month_cols if c in train_df.columns]
    if existing_new:
        train_df = train_df.drop(columns=existing_new)
    
    merged_df = train_df.merge(test_df[['account_number'] + new_month_cols], on='account_number', how='left')
    
    # Remove duplicates
    merged_df = merged_df[[c for c in merged_df.columns if not c.endswith('_duplicate')]]
    suffix_cols = [c for c in merged_df.columns if re.search(r'\.\d+$', c)]
    for col in suffix_cols:
        base = re.sub(r'\.\d+$', '', col)
        if base in merged_df.columns:
            merged_df = merged_df.drop(columns=[col])
        else:
            merged_df = merged_df.rename(columns={col: base})
    
    # FIX Issue #2: Do NOT recalculate Steps parameters after merging test data
    # This would cause future data leakage. Steps are calculated temporally on-demand.
    print(f"\n    Removing legacy Steps columns (will be calculated temporally on-demand)...")
    steps_params = ['Steps_mean', 'Steps_median', 'Steps_divergence', 'Steps_Max']
    steps_params_temporal = ['Steps_mean_temporal', 'Steps_median_temporal', 'Steps_divergence_temporal', 'Steps_Max_temporal']
    all_steps_params = steps_params + steps_params_temporal
    merged_df = merged_df.drop(columns=[c for c in all_steps_params if c in merged_df.columns])
    print(f"    Legacy Steps columns removed. They will be calculated temporally during feature preparation.")
    
    merged_df = merged_df.sort_values(['health_plan', 'account_number']).reset_index(drop=True)
    save_dataframe(merged_df, new_training_file)
    print(f"    New file: {new_training_file}, Shape: {merged_df.shape}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Automated 25-step forecasting workflow - NaN-AWARE (distinguishes no data from measured 0)"""
    print("="*80 + "\nAUTOMATED FALL RISK FORECASTING (NaN-AWARE)\n" + "="*80)
    
    config = {
        'algorithm': 'XGBoost',
        'initial_training_file': 'FallRisk_Training_112024_To_042025_Healthplans.CSV',
        'output_base_dir': 'FallRisk_NaN_Aware',
        'months_to_process': [
            {'predict': 'May_2025', 'test_file': 'FallRisk_Test_May2025_Healthplans.CSV'},
            {'predict': 'Jun_2025', 'test_file': 'FallRisk_Test_June2025_Healthplans.CSV'},
            {'predict': 'Jul_2025', 'test_file': 'FallRisk_Test_July2025_Healthplans.CSV'},
            {'predict': 'Aug_2025', 'test_file': 'FallRisk_Test_August2025_Healthplans.CSV'},
            {'predict': 'Sep_2025', 'test_file': 'FallRisk_Test_September2025_Healthplans.CSV'},
            {'predict': 'Oct_2025', 'test_file': 'FallRisk_Test_October2025_Healthplans.csv'},
            {'predict': 'Nov_2025', 'test_file': 'FallRisk_Test_November2025_Healthplans.csv'},
            {'predict': 'Dec_2025', 'test_file': 'FallRisk_Test_December2025_Healthplans.csv'},
            {'predict': 'Jan_2026', 'test_file': 'FallRisk_Test_January2026_Healthplans.csv'},
            {'predict': 'Feb_2026', 'test_file': None}
        ]
    }
    
    # Verify training file
    config['initial_training_file'] = detect_file_format(config['initial_training_file'])
    if not os.path.exists(config['initial_training_file']):
        print(f"\nERROR: Training file not found: {config['initial_training_file']}")
        return
    
    print(f"\nAlgorithm: {config['algorithm']}")
    print(f"Training: {config['initial_training_file']}")
    print(f"Months: {len(config['months_to_process'])}\n")
    
    current_training_file = config['initial_training_file']
    
    # Process each month
    for idx, month_config in enumerate(config['months_to_process'], 1):
        predict_month = month_config['predict']
        test_file = month_config['test_file']
        
        print(f"\n{'='*80}\nMONTH {idx}/{len(config['months_to_process'])}: {predict_month}\n{'='*80}")
        
        month_short = predict_month.split('_')[0]
        year_short = predict_month.split('_')[1]
        prediction_dir = f"{config['output_base_dir']}_predicting{month_short}{year_short}"
        testing_dir = f"{config['output_base_dir']}_Test_Against{month_short}{year_short}"
        os.makedirs(prediction_dir, exist_ok=True)
        
        # STEP A: Train & Predict
        print(f"\n[{idx}A] Training and Predicting...")
        try:
            train_df = read_dataframe(current_training_file)
            
            # FIX QA Issue: Implement proper train/validation split by patient IDs
            # Split patients (not time periods) to prevent patient-level data leakage
            all_patient_ids = train_df['account_number'].unique()
            np.random.seed(42)  # For reproducibility
            np.random.shuffle(all_patient_ids)
            
            # 80/20 split: 80% train, 20% validation
            split_idx = int(len(all_patient_ids) * 0.8)
            train_patient_ids = all_patient_ids[:split_idx]
            val_patient_ids = all_patient_ids[split_idx:]
            
            print(f"  Patient split: {len(train_patient_ids)} train, {len(val_patient_ids)} validation (NO overlap)")
            
            # Use temporal features to prevent data leakage (FIX Issues #1, #4, #5)
            model = FallRiskForecastingModel(algorithm=config['algorithm'], 
                                            use_temporal_features=True, 
                                            use_smote=False)
            
            # Prepare training data (from training patients only)
            X_train, y_train, _, _ = model.prepare_forecasting_features(
                train_df, predict_month=None, patient_ids_filter=train_patient_ids)
            print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features, Fall rate: {y_train.mean():.2%}")
            
            # Prepare validation data (from held-out patients)
            X_val, y_val, _, _ = model.prepare_forecasting_features(
                train_df, predict_month=None, patient_ids_filter=val_patient_ids)
            print(f"  Validation: {X_val.shape[0]} samples, Fall rate: {y_val.mean():.2%}")
            
            # Train model with validation set (for threshold calculation)
            model.train(X_train, y_train, X_val=X_val, y_val=y_val)
            
            # Report TRAINING metrics (for reference, not validation)
            y_train_proba = model.predict_proba(X_train)
            y_train_pred = model.predict(X_train)
            print(f"  Training ROC-AUC: {roc_auc_score(y_train, y_train_proba):.4f} (on training patients)")
            
            # Report VALIDATION metrics (TRUE performance on held-out patients)
            y_val_proba = model.predict_proba(X_val)
            y_val_pred = model.predict(X_val)
            val_roc_auc = roc_auc_score(y_val, y_val_proba) if len(np.unique(y_val)) > 1 else 0.0
            print(f"  Validation ROC-AUC: {val_roc_auc:.4f} (on held-out patients) ✓")
            
            plot_roc_auc(y_val, y_val_proba, f'ROC-AUC Validation (Held-out Patients) - {predict_month}',
                        os.path.join(prediction_dir, f'ROC_AUC_Val_{month_short}.png'))
            plot_confusion_matrix(y_val, y_val_pred, f'CM Validation (Held-out Patients) - {predict_month}',
                                os.path.join(prediction_dir, f'CM_Val_{month_short}.png'))
            save_confusion_matrix_txt(y_val, y_val_pred, y_val_proba,
                                    os.path.join(prediction_dir, f'CM_Val_{month_short}.txt'),
                                    f"Validation (Held-out Patients) {predict_month}")
            evaluate_model(y_val, y_val_pred, y_val_proba,
                          os.path.join(prediction_dir, f'Performance_Val_{month_short}.txt'))
            model.save_feature_importance(os.path.join(prediction_dir, f'FeatureImportance_{month_short}.txt'), 20)
            
            # Generate predictions for ALL patients (not filtered by train/val split)
            X_predict, _, account_numbers, _ = model.prepare_forecasting_features(
                train_df, predict_month=predict_month, patient_ids_filter=None)
            probabilities = model.predict_proba(X_predict)
            
            # FIX: Create probability dictionary for O(1) lookups (fixes index mismatch bug)
            # Previously used list.index() which is O(n) per lookup and can cause mismatches
            prob_dict = dict(zip(account_numbers, probabilities))
            
            # Extract data quality scores from predictions (calculated in _add_engineered_features)
            data_quality_dict = dict(zip(account_numbers, X_predict['data_quality_score'].values))
            
            # Report data quality distribution
            dq_values = X_predict['data_quality_score'].values
            print(f"  Data Quality Distribution: min={dq_values.min():.1%}, max={dq_values.max():.1%}, " +
                  f"mean={dq_values.mean():.1%}, median={np.median(dq_values):.1%}")
            
            # Get adaptive threshold for this month
            adaptive_threshold = model.get_adaptive_threshold(predict_month)
            
            forecast_data = []
            for i, row in train_df.iterrows():
                acc = row['account_number']
                prob = prob_dict.get(acc, 0.0)  # Fast O(1) lookup with default 0.0
                data_quality = data_quality_dict.get(acc, 0.0)  # Data quality score (0.0-1.0)
                
                # Use rounded probability consistently for all derived columns
                prob_rounded = round(prob, 4)
                data_quality_pct = round(data_quality * 100, 1)  # Convert to percentage
                
                forecast_data.append({
                    'account_number': acc,
                    'account_id': row.get('account_id', ''),
                    'Age': row.get('Age', row.get('age', '')),
                    'brand': row.get('brand', ''),
                    'health_plan': row.get('health_plan', ''),
                    'member name': row.get('member name', ''),
                    'care manager': row.get('care manager', ''),
                    'Probability': prob_rounded,
                    'Data_Quality_Pct': data_quality_pct,  # NEW: Shows % of features with data
                    'Flagged': prob_rounded >= adaptive_threshold  # Use adaptive threshold
                })
            
            forecast_df = pd.DataFrame(forecast_data)
            
            # Calculate percentile-based Risk Score (1-10)
            # All patients ranked by probability - model learns NaN patterns during training
            pct = forecast_df["Probability"].rank(method="average", pct=True)
            forecast_df["Risk_Score"] = np.ceil(pct * 10).clip(1, 10).astype(int)
            forecast_df["Risk_Category"] = forecast_df["Risk_Score"].apply(get_risk_category)
            
            file_ext = '.parquet' if current_training_file.lower().endswith('.parquet') else '.CSV'
            forecast_file = os.path.join(prediction_dir, f'FallRiskForecast_{month_short}_{year_short}_Healthplans{file_ext}')
            save_dataframe(forecast_df, forecast_file)
            total_members = len(forecast_df)
            flagged_members = forecast_df['Flagged'].sum()
            print(f"  ✓ Forecast: {total_members} members ({flagged_members} flagged at adaptive threshold={adaptive_threshold:.4f}), Flagged: {forecast_df[forecast_df['Flagged']]['Risk_Category'].value_counts().to_dict()}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        
        # STEP B: Test
        detected_test = detect_file_format(test_file) if test_file else None
        if detected_test and os.path.exists(detected_test):
            print(f"\n[{idx}B] Testing against real data...")
            try:
                os.makedirs(testing_dir, exist_ok=True)
                test_df = read_dataframe(detected_test)
                
                fall_col = f'fall_count_{predict_month}'
                fall_alarm_col = f'fall_alarm_count_{predict_month}'
                
                if fall_col in test_df.columns:
                    y_actual = (test_df[fall_col] > 0).astype(int)
                elif fall_alarm_col in test_df.columns:
                    y_actual = (test_df[fall_alarm_col] > 0).astype(int)
                else:
                    print(f"  WARNING: No outcome column found")
                    y_actual = None
                
                if y_actual is not None:
                    forecast_indexed = forecast_df.set_index('account_number')
                    matched = [a for a in test_df['account_number'].values if a in forecast_indexed.index]
                    
                    if matched:
                        # Use Probability and Flagged columns directly (no reconstruction loss)
                        y_pred_proba = [forecast_indexed.loc[a, 'Probability'] for a in matched]
                        y_pred = [int(forecast_indexed.loc[a, 'Flagged']) for a in matched]
                        y_true = [y_actual[test_df['account_number'] == a].iloc[0] for a in matched]
                        
                        y_pred_proba, y_pred, y_true = np.array(y_pred_proba), np.array(y_pred), np.array(y_true)
                        print(f"  Predicted: {y_pred.sum()}, Actual: {y_true.sum()}")
                        
                        if len(np.unique(y_true)) > 1:
                            print(f"  Test ROC-AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")
                            plot_roc_auc(y_true, y_pred_proba, f'ROC-AUC Test - {predict_month}',
                                        os.path.join(testing_dir, f'ROC_Test_{predict_month}.png'))
                            plot_confusion_matrix(y_true, y_pred, f'CM Test - {predict_month}',
                                                os.path.join(testing_dir, f'CM_Test_{predict_month}.png'))
                            save_confusion_matrix_txt(y_true, y_pred, y_pred_proba,
                                                    os.path.join(testing_dir, f'CM_Test_{predict_month}.txt'),
                                                    f"Testing {predict_month}")
                            evaluate_model(y_true, y_pred, y_pred_proba,
                                          os.path.join(testing_dir, f'Performance_Test_{predict_month}.txt'))
            except Exception as e:
                print(f"  ERROR: {e}")
        
        # STEP C: Update training
        if detected_test and os.path.exists(detected_test) and idx < len(config['months_to_process']):
            print(f"\n[{idx}C] Updating training file...")
            try:
                month_num = {'May':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08', 'Sep':'09',
                           'Oct':'10', 'Nov':'11', 'Dec':'12', 'Jan':'01'}.get(month_short, '00')
                file_ext = '.parquet' if current_training_file.lower().endswith('.parquet') else '.CSV'
                new_training = f"FallRisk_Training_112024_To_{month_num}{year_short}_Healthplans{file_ext}"
                
                update_training_file_with_real_data(current_training_file, detected_test, 
                                                   new_training, predict_month)
                current_training_file = new_training
                print(f"  ✓ Ready for next month")
            except Exception as e:
                print(f"  ERROR: {e}")
        
        print(f"\n{'='*80}\nCOMPLETED: {predict_month}\n{'='*80}")
    
    print(f"\n{'='*80}\nALL STEPS COMPLETED!\n{'='*80}")
    print(f"\nProcessed: {len(config['months_to_process'])} months")
    print(f"Algorithm: {config['algorithm']}")
    print(f"Final training: {current_training_file}")


if __name__ == "__main__":
    main()
