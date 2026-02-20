"""
Fall Risk Forecasting Model - NaN-Aware Beta Version
Predicts next month falls using current month features with no data leakage
"""

import pandas as pd
import numpy as np
import os, pickle, warnings, re
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, classification_report, 
                             accuracy_score, precision_score, recall_score, f1_score, fbeta_score)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
import xgboost as xgb

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    import pyarrow
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_palette("husl")
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
    HAS_SEABORN = False


# ============================================================================
# FILE I/O
# ============================================================================

def detect_file_format(file_path_base):
    """Detect CSV or Parquet file"""
    base = file_path_base.rsplit('.', 1)[0] if '.' in file_path_base else file_path_base
    for ext in ['.CSV', '.csv', '.parquet']:
        if os.path.exists(base + ext):
            return base + ext
    return file_path_base


def read_dataframe(file_path):
    """Read DataFrame from CSV or Parquet"""
    actual_path = detect_file_format(file_path)
    if not os.path.exists(actual_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if actual_path.lower().endswith('.parquet'):
        if not HAS_PARQUET:
            raise ImportError("Parquet file found but pyarrow not installed")
        print(f"  Reading Parquet: {actual_path}")
        return pd.read_parquet(actual_path)
    print(f"  Reading CSV: {actual_path}")
    return pd.read_csv(actual_path)


def save_dataframe(df, file_path):
    """Save DataFrame to CSV or Parquet"""
    if file_path.lower().endswith('.parquet') and HAS_PARQUET:
        print(f"  Saving Parquet: {file_path}")
        df.to_parquet(file_path, index=False, engine='pyarrow')
    else:
        print(f"  Saving CSV: {file_path}")
        df.to_csv(file_path.replace('.parquet', '.CSV'), index=False)


# ============================================================================
# MODEL CLASS
# ============================================================================

class FallRiskForecastingModel:
    """Fall Risk Forecasting Model with adaptive thresholds and NaN-aware features"""
    
    def __init__(self, algorithm='XGBoost', use_temporal_features=True, use_smote=None):
        self.algorithm = algorithm
        self.use_temporal_features = use_temporal_features
        self.use_smote = use_smote if use_smote is not None else (not use_temporal_features)
        self.model = None
        self.calibrated_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.use_scaling = algorithm == 'LogisticRegression'
        self.optimal_threshold = 0.08  # Default threshold (will be optimized for 70% recall target)
        self.adaptive_thresholds = {}  # Will store recall-optimized thresholds per month
        
    def _create_model(self, class_weights=None):
        """Create model instance"""
        if self.algorithm == 'XGBoost':
            scale_pos_weight = 30.0  # Increased to 30 to better emphasize minority class (falls)
            if class_weights is not None and len(class_weights) == 2:
                scale_pos_weight = 30.0  # Fixed at 30.0
            return xgb.XGBClassifier(
                n_estimators=800, max_depth=4, learning_rate=0.01, subsample=0.9,
                colsample_bytree=0.9, colsample_bylevel=0.9, scale_pos_weight=scale_pos_weight,
                min_child_weight=1, reg_alpha=0.05, reg_lambda=0.5,
                random_state=42, eval_metric='logloss', use_label_encoder=False,
                tree_method='hist', max_bin=512, min_split_loss=0.05,
                early_stopping_rounds=100,  # Stop if no improvement for 100 rounds
                missing=np.nan  # XGBoost handles NaN natively
            )
        elif self.algorithm == 'RandomForest':
            return RandomForestClassifier(
                n_estimators=800, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                max_features='sqrt', class_weight='balanced_subsample', criterion='entropy',
                bootstrap=True, oob_score=True, random_state=42, n_jobs=-1
            )
        elif self.algorithm == 'GradientBoosting':
            return GradientBoostingClassifier(n_estimators=200, max_depth=7, learning_rate=0.05,
                                             subsample=0.8, random_state=42)
        elif self.algorithm == 'LogisticRegression':
            return LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced',
                                     C=0.1, solver='liblinear')
        raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def prepare_forecasting_features(self, df, predict_month=None, patient_ids_filter=None):
        """Prepare features for forecasting (Month N → Month N+1) with no data leakage"""
        month_order = ['11_2024', '12_2024', '01_2025', '02_2025', '03_2025', '04_2025',
                      '05_2025', '06_2025', '07_2025', '08_2025', '09_2025', '10_2025',
                      '11_2025', '12_2025', '01_2026', '02_2026']
        
        id_cols = [col for col in ['account_number', 'account_id', 'brand', 'health_plan', 'age', 'Age'] 
                   if col in df.columns]
        
        # Filter by patient IDs (for train/val split - prevents patient-level leakage)
        if patient_ids_filter is not None:
            df = df[df['account_number'].isin(patient_ids_filter)].copy()
            print(f"  Filtered to {len(patient_ids_filter)} patients")
        
        # Get available months
        available_months = sorted(
            list(set(['_'.join(col.split('_')[-2:]) for col in df.columns 
                     if '_' in col and col not in id_cols and len(col.split('_')) >= 3
                     and '_'.join(col.split('_')[-2:]) in month_order])),
            key=lambda x: month_order.index(x)
        )
        print(f"  Available months: {available_months}")
        
        X_list, y_list, account_list = [], [], []
        
        if predict_month:
            # Single prediction mode
            predict_idx = month_order.index(predict_month)
            current_month = month_order[predict_idx - 1] if predict_idx > 0 else None
            if not current_month or current_month not in available_months:
                raise ValueError(f"Cannot predict {predict_month} - no previous month data")
            
            # Calculate historical data quality UP TO current_month (prevents future data leakage)
            historical_months = [m for m in available_months if month_order.index(m) <= month_order.index(current_month)]
            historical_cols = [col for col in df.columns 
                             if any(f'_{m}' in col for m in historical_months) and col not in id_cols
                             and not col.startswith(('fall_count_', 'fall_alarm_count_'))]
            
            df['historical_data_quality'] = df[historical_cols].notna().sum(axis=1) / len(historical_cols) if historical_cols else 0.0
            
            feature_cols = [col for col in df.columns 
                          if col.endswith(f'_{current_month}') and col not in id_cols
                          and not col.startswith(('fall_count_', 'fall_alarm_count_'))]
            
            X = df[feature_cols].copy()
            X.columns = [col.replace(f'_{current_month}', '') for col in X.columns]
            X['age'] = df['age'].values if 'age' in df.columns else df.get('Age', 75)
            X['historical_data_quality'] = df['historical_data_quality'].values
            
            # Use temporal Steps features (only data up to current_month - prevents leakage)
            if self.use_temporal_features:
                df_with_steps = self._calculate_steps_temporal(df, month_order, current_month)
                for param in ['Steps_mean_temporal', 'Steps_median_temporal', 'Steps_divergence_temporal', 'Steps_Max_temporal']:
                    if param in df_with_steps.columns:
                        X[param] = df_with_steps[param].values
            
            target_col = f'fall_count_{predict_month}'
            y = (df[target_col] > 0).astype(int) if target_col in df.columns else None
            account_numbers = df['account_number'].values if 'account_number' in df.columns else None
            
        else:
            # Training mode: all month pairs
            for i in range(len(available_months) - 1):
                current_month = available_months[i]
                next_month = available_months[i + 1]
                print(f"    Pairing: {current_month} → {next_month}")
                
                # Calculate historical data quality UP TO current_month (prevents leakage)
                historical_months = [m for m in available_months if month_order.index(m) <= month_order.index(current_month)]
                historical_cols = [col for col in df.columns 
                                 if any(f'_{m}' in col for m in historical_months) and col not in id_cols
                                 and not col.startswith(('fall_count_', 'fall_alarm_count_'))]
                
                df[f'historical_data_quality_{i}'] = df[historical_cols].notna().sum(axis=1) / len(historical_cols) if historical_cols else 0.0
                
                feature_cols = [col for col in df.columns 
                              if col.endswith(f'_{current_month}') and col not in id_cols
                              and not col.startswith(('fall_count_', 'fall_alarm_count_'))]
                
                if not feature_cols:
                    continue
                
                X_month = df[feature_cols].copy()
                X_month.columns = [col.replace(f'_{current_month}', '') for col in X_month.columns]
                X_month['age'] = df['age'].values if 'age' in df.columns else df.get('Age', 75)
                X_month['historical_data_quality'] = df[f'historical_data_quality_{i}'].values
                
                # Use temporal Steps features (prevents leakage)
                if self.use_temporal_features:
                    df_with_steps = self._calculate_steps_temporal(df, month_order, current_month)
                    for param in ['Steps_mean_temporal', 'Steps_median_temporal', 'Steps_divergence_temporal', 'Steps_Max_temporal']:
                        if param in df_with_steps.columns:
                            X_month[param] = df_with_steps[param].values
                
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
        
        # NaN HANDLING: Keep NaN to distinguish "no data collected" from "measured 0"
        if 'age' in X.columns:
            X['age'] = X['age'].fillna(75)
        
        X = self._add_engineered_features(X)
        return X, y, account_numbers, current_month if predict_month else "multiple_months"
    
    def _calculate_steps_temporal(self, df, month_order, current_month):
        """Calculate Steps statistics using only data UP TO current_month (prevents leakage)"""
        if current_month not in month_order:
            return df
        
        current_idx = month_order.index(current_month)
        steps_cols = sorted(
            [col for col in df.columns if col.startswith('avg_daily_steps_')
             and any(f'_{m}' in col for m in month_order[:current_idx + 1])],
            key=lambda c: next((month_order.index(m) for m in month_order if f'_{m}' in c), 999)
        )
        
        if not steps_cols:
            df['Steps_mean_temporal'] = df['Steps_median_temporal'] = df['Steps_divergence_temporal'] = df['Steps_Max_temporal'] = 0.0
            return df
        
        steps_data = df[steps_cols].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df['Steps_mean_temporal'] = np.nanmean(steps_data, axis=1)
            df['Steps_median_temporal'] = np.nanmedian(steps_data, axis=1)
            df['Steps_divergence_temporal'] = np.nanstd(steps_data, axis=1)
            df['Steps_Max_temporal'] = np.nanmax(steps_data, axis=1)
        return df
    
    def _add_engineered_features(self, X):
        """Add engineered features (NaN-aware: distinguishes no data from measured 0)"""
        X = X.copy()
        
        X['data_quality_score'] = X.get('historical_data_quality', 
            X[[col for col in X.columns if col != 'age']].notna().sum(axis=1) / max(len([col for col in X.columns if col != 'age']), 1))
        X['very_low_data_flag'] = (X['data_quality_score'] < 0.10).astype(int)
        
        steps_temporal = [col for col in X.columns if 'Steps' in col and 'temporal' in col]
        if steps_temporal:
            X['has_activity_data'] = ((X[steps_temporal].notna()) & (X[steps_temporal].abs() > 0)).any(axis=1).astype(int)
        
        if 'age' in X.columns:
            if 'avg_daily_steps' in X.columns:
                X['age_x_steps'] = np.where(X['avg_daily_steps'].notna(), X['age'] * X['avg_daily_steps'], np.nan)
                X['age_x_steps_ratio'] = np.where((X['avg_daily_steps'].notna()) & (X['avg_daily_steps'] > 0),
                                                  X['age'] / (X['avg_daily_steps'] + 1), np.nan)
            for col in ['fall_alarm_count', 'assist_count']:
                if col in X.columns:
                    X[f'age_x_{col.split("_")[0]}'] = np.where(X[col].notna(), X['age'] * X[col], np.nan)
        
        if 'avg_daily_steps' in X.columns:
            X['steps_squared'] = np.where(X['avg_daily_steps'].notna(), X['avg_daily_steps'] ** 2, np.nan)
            X['low_activity_flag'] = np.where((X['avg_daily_steps'].notna()) & (X['avg_daily_steps'] < 1000), 1, 0)
            X['missing_activity_data'] = X['avg_daily_steps'].isna().astype(int)
        
        fall_cols = [col for col in X.columns if 'fall_' in col or 'assist' in col or 'dispatch' in col]
        if fall_cols:
            X['total_fall_events'] = X[fall_cols].sum(axis=1, skipna=True)
        
        if 'er_dispatch_count' in X.columns:
            X['has_er_dispatch'] = ((X['er_dispatch_count'].notna()) & (X['er_dispatch_count'] > 0)).astype(int)
        if 'sentiment_negative_count' in X.columns:
            X['has_negative_sentiment'] = ((X['sentiment_negative_count'].notna()) & (X['sentiment_negative_count'] > 0)).astype(int)
        if 'fall_alarm_count' in X.columns and 'assist_count' in X.columns:
            X['fall_to_assist_ratio'] = np.where((X['fall_alarm_count'].notna()) & (X['assist_count'].notna()),
                                                 X['fall_alarm_count'] / (X['assist_count'] + 1), np.nan)
        if 'button_press_count' in X.columns and 'fall_alarm_count' in X.columns:
            X['button_to_fall_ratio'] = np.where((X['button_press_count'].notna()) & (X['fall_alarm_count'].notna()),
                                                 X['button_press_count'] / (X['fall_alarm_count'] + 1), np.nan)
        return X
    
    def _calculate_optimal_threshold(self, X, y, method='recall_target', target_recall=0.70):
        """Calculate optimal threshold to achieve target recall while maximizing precision"""
        if len(np.unique(y)) < 2:
            print(f"    WARNING: Only one class in validation, using default threshold")
            return self.optimal_threshold
        
        # Get predicted probabilities
        y_proba = self.predict_proba(X)
        
        # Use a wider, lower range to catch more falls
        p_min, p_max = np.percentile(y_proba, [0.1, 95])
        thresholds = np.linspace(max(p_min, 0.001), min(p_max, 0.5), 300)
        
        print(f"    Searching {len(thresholds)} thresholds between {p_min:.4f} and {p_max:.4f}")
        print(f"    Strategy: Find threshold achieving recall >= {target_recall:.1%}, then maximize precision")
        
        # First pass: Find all thresholds meeting target recall
        valid_thresholds = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            if y_pred.sum() == 0:
                continue
            
            recall = recall_score(y, y_pred, zero_division=0)
            precision = precision_score(y, y_pred, zero_division=0)
            
            # Keep thresholds that meet or exceed target recall
            if recall >= target_recall:
                valid_thresholds.append({
                    'threshold': threshold,
                    'recall': recall,
                    'precision': precision,
                    'f2': fbeta_score(y, y_pred, beta=2.0, zero_division=0)
                })
        
        # If no threshold meets target, find threshold with highest recall
        if not valid_thresholds:
            print(f"    WARNING: No threshold achieves {target_recall:.1%} recall, finding threshold with maximum recall")
            best_recall = -1
            best_threshold = self.optimal_threshold
            
            for threshold in thresholds:
                y_pred = (y_proba >= threshold).astype(int)
                if y_pred.sum() == 0:
                    continue
                recall = recall_score(y, y_pred, zero_division=0)
                if recall > best_recall:
                    best_recall = recall
                    best_threshold = threshold
        else:
            # Among valid thresholds, choose one with best precision (or F2 if precision similar)
            print(f"    Found {len(valid_thresholds)} thresholds meeting recall >= {target_recall:.1%}")
            # Sort by precision descending
            valid_thresholds.sort(key=lambda x: x['precision'], reverse=True)
            best_threshold = valid_thresholds[0]['threshold']
        
        # Validate the best threshold found
        y_pred_best = (y_proba >= best_threshold).astype(int)
        final_precision = precision_score(y, y_pred_best, zero_division=0)
        final_recall = recall_score(y, y_pred_best, zero_division=0)
        final_f2 = fbeta_score(y, y_pred_best, beta=2.0, zero_division=0)
        
        print(f"    Optimal threshold: {best_threshold:.4f} (Recall={final_recall:.4f}, Precision={final_precision:.4f}, F2={final_f2:.4f})")
        
        return best_threshold
    
    def train(self, X_train, y_train, X_val=None, y_val=None, sample_weights=None, predict_month=None):
        """Train model with proper validation (prevents patient-level leakage)"""
        self.feature_names = list(X_train.columns)
        
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        if len(class_weights) == 2:
            class_weights[1] *= 9.5  # Boost minority class for better recall
        
        print(f"    Class distribution: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"    Class weights: {dict(zip(classes, class_weights))}")
        
        # Calculate sample weights (prioritize complete data)
        if sample_weights is None:
            quality_col = 'historical_data_quality' if 'historical_data_quality' in X_train.columns else 'data_quality_score'
            if quality_col in X_train.columns:
                completeness = X_train[quality_col].values
            else:
                feature_cols = [col for col in X_train.columns if col not in ['age', 'data_quality_score', 'historical_data_quality']]
                completeness = X_train[feature_cols].notna().sum(axis=1) / len(feature_cols)
            sample_weights = 1.0 + completeness
            print(f"    Sample weights: min={sample_weights.min():.2f}, max={sample_weights.max():.2f}, mean={sample_weights.mean():.2f}")
        
        # SMOTE disabled to prevent temporal mixing (maintains temporal integrity)
        if not self.use_smote:
            print(f"    SMOTE disabled (prevents temporal mixing)")
        
        X_train_scaled = self.scaler.fit_transform(X_train) if self.use_scaling else X_train
        if self.use_scaling:
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        # Prepare validation data for early stopping
        if X_val is not None and y_val is not None and self.algorithm == 'XGBoost':
            X_val_scaled = self.scaler.transform(X_val) if self.use_scaling else X_val
            if self.use_scaling:
                X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
        
        self.model = self._create_model(class_weights if self.algorithm == 'XGBoost' else None)
        
        # Fit with early stopping if validation data available
        if X_val is not None and y_val is not None and self.algorithm == 'XGBoost':
            self.model.fit(
                X_train_scaled, y_train, 
                sample_weight=sample_weights,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
            print(f"    Early stopping: Best iteration = {self.model.best_iteration}")
        else:
            self.model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        
        # Calibration disabled - XGBoost probabilities are already well-calibrated
        self.calibrated_model = None
        print(f"    Calibration: DISABLED (using native XGBoost probabilities)")
        
        # Calculate optimal threshold using validation data (target 70% recall)
        if X_val is not None and y_val is not None:
            print(f"    Calculating optimal threshold using recall-focused strategy...")
            optimal_threshold = self._calculate_optimal_threshold(X_val, y_val, method='recall_target', target_recall=0.70)
            
            # Store in adaptive thresholds if predict_month is provided
            if predict_month:
                self.adaptive_thresholds[predict_month] = optimal_threshold
                print(f"    Stored adaptive threshold for {predict_month}: {optimal_threshold:.4f}")
            else:
                # Update default threshold if no specific month
                self.optimal_threshold = optimal_threshold
        
        print(f"  Model trained: {self.algorithm}")
    
    def predict_proba(self, X):
        """Predict probabilities"""
        X_aligned = X[self.feature_names].copy()
        if self.use_scaling:
            X_aligned = X_aligned.fillna(0)
        X_scaled = self.scaler.transform(X_aligned) if self.use_scaling else X_aligned
        if self.use_scaling:
            X_scaled = pd.DataFrame(X_scaled, columns=X_aligned.columns)
        proba = self.calibrated_model.predict_proba(X_scaled) if self.calibrated_model else self.model.predict_proba(X_scaled)
        return proba[:, 1]
    
    def get_adaptive_threshold(self, predict_month):
        """Get adaptive threshold based on prediction month (recall-optimized)"""
        threshold = self.adaptive_thresholds.get(predict_month, self.optimal_threshold)
        threshold_type = "recall-optimized" if predict_month in self.adaptive_thresholds else "default"
        print(f"    Using {threshold_type} threshold for {predict_month}: {threshold:.4f}")
        return threshold
    
    def predict(self, X, threshold=None, predict_month=None):
        """Predict classes with optional adaptive threshold"""
        if threshold is None:
            threshold = self.get_adaptive_threshold(predict_month) if predict_month else self.optimal_threshold
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
            for idx, (name, desc) in enumerate([('gain', 'Gain'), ('weight', 'Weight'), ('cover', 'Cover')], 1):
                if name in importance_dict:
                    f.write("="*80 + f"\n{idx}. {desc.upper()}\n" + "="*80 + "\n")
                    f.write(f"{'Rank':<6} {'Feature':<50} {'Score':>15}\n" + "-"*80 + "\n")
                    for rank, (feat, score) in enumerate(importance_dict[name], 1):
                        f.write(f"{rank:<6} {feat:<50} {score:>15.4f}\n")
                    f.write("\n")
    
    def save(self, filepath):
        """Save model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model, 'calibrated_model': self.calibrated_model, 'scaler': self.scaler,
                'feature_names': self.feature_names, 'algorithm': self.algorithm, 'use_scaling': self.use_scaling,
                'optimal_threshold': self.optimal_threshold, 'adaptive_thresholds': self.adaptive_thresholds,
                'use_temporal_features': self.use_temporal_features, 'use_smote': self.use_smote
            }, f)
    
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

def get_risk_category(risk_score):
    """Get risk category from score"""
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
    print(f"  Accuracy: {metrics['Accuracy']:.4f}, ROC-AUC: {metrics['ROC-AUC']:.4f}")


def update_training_file(old_file, test_file, new_file, target_month):
    """Update training file with new month data (prevents leakage by removing Steps columns)"""
    print(f"\n  Updating training file with {target_month} data...")
    train_df = read_dataframe(old_file)
    test_df = read_dataframe(test_file)
    
    id_cols = [c for c in ['account_number', 'account_id', 'brand', 'health_plan', 'age', 'Age'] 
               if c in train_df.columns]
    new_month_cols = [c for c in test_df.columns if c not in id_cols and target_month in c]
    
    # Merge and clean
    existing_new = [c for c in new_month_cols if c in train_df.columns]
    if existing_new:
        train_df = train_df.drop(columns=existing_new)
    merged_df = train_df.merge(test_df[['account_number'] + new_month_cols], on='account_number', how='left')
    merged_df = merged_df[[c for c in merged_df.columns if not c.endswith('_duplicate')]]
    
    # Remove duplicates with numeric suffixes
    suffix_cols = [c for c in merged_df.columns if re.search(r'\.\d+$', c)]
    for col in suffix_cols:
        base = re.sub(r'\.\d+$', '', col)
        merged_df = merged_df.drop(columns=[col]) if base in merged_df.columns else merged_df.rename(columns={col: base})
    
    # Remove all Steps columns (will be recalculated temporally on-demand - prevents leakage)
    steps_cols = [c for c in merged_df.columns if c.startswith('Steps_')]
    merged_df = merged_df.drop(columns=steps_cols)
    
    merged_df = merged_df.sort_values(['health_plan', 'account_number']).reset_index(drop=True)
    save_dataframe(merged_df, new_file)
    print(f"    New file: {new_file}, Shape: {merged_df.shape}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Automated fall risk forecasting workflow - NaN-aware, no data leakage"""
    print("="*80 + "\nFALL RISK FORECASTING - BETA VERSION\n" + "="*80)
    
    config = {
        'algorithm': 'XGBoost',
        'initial_training_file': 'FallRisk_Training_112024_To_042025_Healthplans',
        'months_to_process': [
            {'predict': '05_2025', 'test_file': 'FallRisk_Test_05_2025_Healthplans'},
            {'predict': '06_2025', 'test_file': 'FallRisk_Test_06_2025_Healthplans'},
            {'predict': '07_2025', 'test_file': 'FallRisk_Test_07_2025_Healthplans'},
            {'predict': '08_2025', 'test_file': 'FallRisk_Test_08_2025_Healthplans'},
            {'predict': '09_2025', 'test_file': 'FallRisk_Test_09_2025_Healthplans'},
            {'predict': '10_2025', 'test_file': 'FallRisk_Test_10_2025_Healthplans'},
            {'predict': '11_2025', 'test_file': 'FallRisk_Test_11_2025_Healthplans'},
            {'predict': '12_2025', 'test_file': 'FallRisk_Test_12_2025_Healthplans'},
            {'predict': '01_2026', 'test_file': 'FallRisk_Test_01_2026_Healthplans'},
            {'predict': '02_2026', 'test_file': None}
        ]
    }
    
    # Verify training file
    config['initial_training_file'] = detect_file_format(config['initial_training_file'])
    if not os.path.exists(config['initial_training_file']):
        print(f"\nERROR: Training file not found: {config['initial_training_file']}")
        return
    
    print(f"\nAlgorithm: {config['algorithm']}")
    print(f"Training: {config['initial_training_file']}")
    print(f"Months to process: {len(config['months_to_process'])}\n")
    
    current_training_file = config['initial_training_file']
    
    # Process each month
    for idx, month_config in enumerate(config['months_to_process'], 1):
        predict_month = month_config['predict']
        test_file = month_config['test_file']
        
        print(f"\n{'='*80}\nMONTH {idx}/{len(config['months_to_process'])}: {predict_month}\n{'='*80}")
        
        # Month is already in digit format (MM_YYYY)
        mm, yyyy = predict_month.split('_')
        mm_yyyy = f"{mm}{yyyy}"
        
        # NEW FOLDER NAMING: FallRisk_BetaVersion_Predict_MMYYYY and FallRisk_BetaVersion_Test_Including_MMYYYY
        prediction_dir = f"FallRisk_BetaVersion_Predict_{mm_yyyy}"
        testing_dir = f"FallRisk_BetaVersion_Test_Including_{mm_yyyy}"
        os.makedirs(prediction_dir, exist_ok=True)
        
        # STEP A: Train & Predict
        print(f"\n[{idx}A] Training and Predicting...")
        try:
            train_df = read_dataframe(current_training_file)
            
            # Split patients by ID with stratification (prevents patient-level leakage - QA FIX #1)
            # Stratify based on whether patient has ever had a fall event
            all_patient_ids = train_df['account_number'].unique()
            fall_count_cols = [c for c in train_df.columns if 'fall_count_' in c]
            patient_fall_status = train_df.groupby('account_number').apply(
                lambda x: (x[fall_count_cols].sum().sum() > 0) if fall_count_cols else False
            )
            
            # Use StratifiedShuffleSplit to ensure balanced representation of fall events
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(splitter.split(all_patient_ids, patient_fall_status[all_patient_ids]))
            train_patient_ids = all_patient_ids[train_idx]
            val_patient_ids = all_patient_ids[val_idx]
            
            # Print stratification statistics
            train_fall_rate = patient_fall_status[train_patient_ids].mean()
            val_fall_rate = patient_fall_status[val_patient_ids].mean()
            print(f"  Patient split: {len(train_patient_ids)} train, {len(val_patient_ids)} validation (NO overlap)")
            print(f"  Stratification: Train fall rate={train_fall_rate:.2%}, Val fall rate={val_fall_rate:.2%}")
            
            # Create model (use_temporal_features=True prevents data leakage - QA FIX)
            model = FallRiskForecastingModel(algorithm=config['algorithm'], use_temporal_features=True, use_smote=False)
            
            # Prepare data
            X_train, y_train, _, _ = model.prepare_forecasting_features(train_df, predict_month=None, patient_ids_filter=train_patient_ids)
            X_val, y_val, _, _ = model.prepare_forecasting_features(train_df, predict_month=None, patient_ids_filter=val_patient_ids)
            print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features, Fall rate: {y_train.mean():.2%}")
            print(f"  Validation: {X_val.shape[0]} samples, Fall rate: {y_val.mean():.2%}")
            
            # QA Fix #3: Minimum validation size check
            min_positive_samples = 30
            val_positive_count = y_val.sum()
            if val_positive_count < min_positive_samples:
                print(f"  ⚠️  WARNING: Validation set has only {val_positive_count} positive samples (minimum recommended: {min_positive_samples})")
                print(f"      This may lead to unreliable validation metrics and unstable threshold optimization")
                print(f"      Consider: (1) Increasing training data, (2) Using smaller validation split, or (3) Accepting higher variance")
            else:
                print(f"  ✓ Validation positive samples: {val_positive_count} (sufficient for reliable metrics)")
            
            # Train model (with optimal threshold calculation)
            model.train(X_train, y_train, X_val=X_val, y_val=y_val, predict_month=predict_month)
            
            # Validation metrics
            y_val_proba = model.predict_proba(X_val)
            y_val_pred = model.predict(X_val)
            val_roc_auc = roc_auc_score(y_val, y_val_proba) if len(np.unique(y_val)) > 1 else 0.0
            print(f"  Validation ROC-AUC: {val_roc_auc:.4f} (on held-out patients)")
            
            # Save validation plots and reports
            plot_roc_auc(y_val, y_val_proba, f'ROC-AUC Validation - {predict_month}',
                        os.path.join(prediction_dir, f'ROC_AUC_Val.png'))
            plot_confusion_matrix(y_val, y_val_pred, f'CM Validation - {predict_month}',
                                os.path.join(prediction_dir, f'CM_Val.png'))
            save_confusion_matrix_txt(y_val, y_val_pred, y_val_proba,
                                    os.path.join(prediction_dir, f'CM_Val.txt'), f"Validation {predict_month}")
            evaluate_model(y_val, y_val_pred, y_val_proba, os.path.join(prediction_dir, f'Performance_Val.txt'))
            model.save_feature_importance(os.path.join(prediction_dir, f'FeatureImportance.txt'), 20)
            
            # Generate predictions for ALL patients
            X_predict, _, account_numbers, _ = model.prepare_forecasting_features(train_df, predict_month=predict_month, patient_ids_filter=None)
            probabilities = model.predict_proba(X_predict)
            prob_dict = dict(zip(account_numbers, probabilities))
            data_quality_dict = dict(zip(account_numbers, X_predict['data_quality_score'].values))
            
            # Get adaptive threshold
            adaptive_threshold = model.get_adaptive_threshold(predict_month)
            
            # Create forecast dataframe
            forecast_data = []
            for i, row in train_df.iterrows():
                acc = row['account_number']
                prob = prob_dict.get(acc, 0.0)  # Keep raw probability
                data_quality = data_quality_dict.get(acc, 0.0)
                
                forecast_data.append({
                    'account_number': acc,
                    'account_id': row.get('account_id', ''),
                    'Age': row.get('Age', row.get('age', '')),
                    'brand': row.get('brand', ''),
                    'health_plan': row.get('health_plan', ''),
                    'member name': row.get('member name', ''),
                    'care manager': row.get('care manager', ''),
                    'Probability': round(prob, 4),  # Round for display only
                    'Probability_Raw': prob,  # Store raw for logic
                    'Data_Quality_Pct': round(data_quality * 100, 1),
                    'Flagged': prob >= adaptive_threshold  # Use raw probability for flagging
                })
            
            forecast_df = pd.DataFrame(forecast_data)
            
            # Calculate adaptive Risk Score (1-10) relative to threshold
            # This creates flexibility as threshold changes month-to-month
            def assign_risk_score_relative(prob, threshold):
                """
                Risk score scaled relative to adaptive threshold
                Score 5 = at threshold (flagged), higher scores above, lower below
                """
                if prob >= threshold * 5.0:    return 10  # 5x threshold - very high risk
                elif prob >= threshold * 3.5:  return 9   # 3.5x threshold
                elif prob >= threshold * 2.5:  return 8   # 2.5x threshold
                elif prob >= threshold * 1.75: return 7   # 1.75x threshold
                elif prob >= threshold * 1.25: return 6   # 1.25x threshold - moderate-high
                elif prob >= threshold * 1.0:  return 5   # AT threshold - flagged (moderate)
                elif prob >= threshold * 0.75: return 4   # 0.75x threshold - below flag
                elif prob >= threshold * 0.5:  return 3   # 0.5x threshold
                elif prob >= threshold * 0.25: return 2   # 0.25x threshold - low risk
                else:                          return 1   # < 0.25x threshold - very low
            
            # Apply relative risk scoring using raw probability
            forecast_df["Risk_Score"] = forecast_df["Probability_Raw"].apply(
                lambda p: assign_risk_score_relative(p, adaptive_threshold)
            )
            forecast_df["Risk_Category"] = forecast_df["Risk_Score"].apply(get_risk_category)
            
            # Keep Probability_Raw for accurate evaluation metrics (AUC, etc.)
            # Note: 'Probability' is rounded for display, 'Probability_Raw' preserves full precision
            
            # Save forecast with new naming: Fall_Risk_score_MMYYYY
            file_ext = '.parquet' if current_training_file.lower().endswith('.parquet') else '.CSV'
            forecast_file = os.path.join(prediction_dir, f'Fall_Risk_score_{mm_yyyy}{file_ext}')
            save_dataframe(forecast_df, forecast_file)
            print(f"  ✓ Forecast: {len(forecast_df)} members ({forecast_df['Flagged'].sum()} flagged at threshold={adaptive_threshold:.4f})")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # STEP B: Test against actual data
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
                        # Use raw probability for accurate AUC calculation (avoid quantization)
                        y_pred_proba = np.array([forecast_indexed.loc[a, 'Probability_Raw'] for a in matched])
                        y_pred = np.array([int(forecast_indexed.loc[a, 'Flagged']) for a in matched])
                        y_true = np.array([y_actual[test_df['account_number'] == a].iloc[0] for a in matched])
                        
                        print(f"  Predicted: {y_pred.sum()}, Actual: {y_true.sum()}")
                        
                        if len(np.unique(y_true)) > 1:
                            print(f"  Test ROC-AUC: {roc_auc_score(y_true, y_pred_proba):.4f} (using raw probabilities)")
                            plot_roc_auc(y_true, y_pred_proba, f'ROC-AUC Test - {predict_month}',
                                        os.path.join(testing_dir, f'ROC_Test.png'))
                            plot_confusion_matrix(y_true, y_pred, f'CM Test - {predict_month}',
                                                os.path.join(testing_dir, f'CM_Test.png'))
                            save_confusion_matrix_txt(y_true, y_pred, y_pred_proba,
                                                    os.path.join(testing_dir, f'CM_Test.txt'), f"Testing {predict_month}")
                            evaluate_model(y_true, y_pred, y_pred_proba, os.path.join(testing_dir, f'Performance_Test.txt'))
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        # STEP C: Update training file for next month
        if detected_test and os.path.exists(detected_test) and idx < len(config['months_to_process']):
            print(f"\n[{idx}C] Updating training file...")
            try:
                file_ext = '.parquet' if current_training_file.lower().endswith('.parquet') else '.CSV'
                new_training = f"FallRisk_Training_112024_To_{mm_yyyy}_Healthplans{file_ext}"
                update_training_file(current_training_file, detected_test, new_training, predict_month)
                current_training_file = new_training
                print(f"  ✓ Ready for next month")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*80}\nCOMPLETED: {predict_month}\n{'='*80}")
    
    print(f"\n{'='*80}\nALL STEPS COMPLETED!\n{'='*80}")
    print(f"Processed: {len(config['months_to_process'])} months")
    print(f"Algorithm: {config['algorithm']}")
    print(f"Final training file: {current_training_file}")


if __name__ == "__main__":
    main()
