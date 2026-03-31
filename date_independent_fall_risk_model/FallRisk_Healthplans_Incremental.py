"""
Fall Risk Forecasting Model - Incremental Mode (Production)
INCREMENTAL MODE: Process one new month at a time (~10-15 min per month, 90% cost savings)
"""

import pandas as pd
import numpy as np
import os, pickle, warnings, re, argparse, sys, glob
from datetime import datetime
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
        self.optimal_threshold = 0.08
        self.adaptive_thresholds = {}
        
    def _create_model(self, class_weights=None):
        """Create model instance"""
        if self.algorithm == 'XGBoost':
            return xgb.XGBClassifier(
                n_estimators=800, max_depth=4, learning_rate=0.01, subsample=0.9,
                colsample_bytree=0.9, colsample_bylevel=0.9, scale_pos_weight=30.0,
                min_child_weight=1, reg_alpha=0.05, reg_lambda=0.5,
                random_state=42, eval_metric='logloss', use_label_encoder=False,
                tree_method='hist', max_bin=512, min_split_loss=0.05,
                early_stopping_rounds=100, missing=np.nan
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
                      '11_2025', '12_2025', '01_2026', '02_2026', '03_2026']
        
        id_cols = [col for col in ['account_number', 'account_id', 'health_plan', 'age', 'Age', 'brand'] 
                   if col in df.columns]
        
        if patient_ids_filter is not None:
            df = df[df['account_number'].isin(patient_ids_filter)].copy()
            print(f"  Filtered to {len(patient_ids_filter)} patients")
        
        available_months = sorted(
            list(set(['_'.join(col.split('_')[-2:]) for col in df.columns 
                     if '_' in col and col not in id_cols and len(col.split('_')) >= 3
                     and not col.startswith('brand_')
                     and '_'.join(col.split('_')[-2:]) in month_order])),
            key=lambda x: month_order.index(x)
        )
        print(f"  Available months: {available_months}")
        
        X_list, y_list, account_list = [], [], []
        
        if predict_month:
            predict_idx = month_order.index(predict_month)
            current_month = month_order[predict_idx - 1] if predict_idx > 0 else None
            if not current_month or current_month not in available_months:
                raise ValueError(f"Cannot predict {predict_month} - no previous month data")
            
            historical_months = [m for m in available_months if month_order.index(m) <= month_order.index(current_month)]
            historical_cols = [col for col in df.columns 
                             if any(f'_{m}' in col for m in historical_months) and col not in id_cols
                             and not col.startswith(('fall_count_', 'fall_alarm_count_', 'brand_'))]
            
            df['historical_data_quality'] = df[historical_cols].notna().sum(axis=1) / len(historical_cols) if historical_cols else 0.0
            
            feature_cols = [col for col in df.columns 
                          if col.endswith(f'_{current_month}') and col not in id_cols
                          and not col.startswith(('fall_count_', 'fall_alarm_count_', 'brand_'))]
            
            X = df[feature_cols].copy()
            X.columns = [col.replace(f'_{current_month}', '') for col in X.columns]
            X['age'] = df['age'].values if 'age' in df.columns else df.get('Age', 75)
            X['historical_data_quality'] = df['historical_data_quality'].values
            
            if self.use_temporal_features:
                df_with_steps = self._calculate_steps_temporal(df, month_order, current_month)
                for param in ['Steps_mean_temporal', 'Steps_median_temporal', 'Steps_divergence_temporal', 'Steps_Max_temporal']:
                    if param in df_with_steps.columns:
                        X[param] = df_with_steps[param].values
            
            target_col = f'fall_count_{predict_month}'
            y = (df[target_col] > 0).astype(int) if target_col in df.columns else None
            account_numbers = df['account_number'].values if 'account_number' in df.columns else None
            
        else:
            for i in range(len(available_months) - 1):
                current_month = available_months[i]
                next_month = available_months[i + 1]
                print(f"    Pairing: {current_month} → {next_month}")
                
                historical_months = [m for m in available_months if month_order.index(m) <= month_order.index(current_month)]
                historical_cols = [col for col in df.columns 
                                 if any(f'_{m}' in col for m in historical_months) and col not in id_cols
                                 and not col.startswith(('fall_count_', 'fall_alarm_count_', 'brand_'))]
                
                df[f'historical_data_quality_{i}'] = df[historical_cols].notna().sum(axis=1) / len(historical_cols) if historical_cols else 0.0
                
                feature_cols = [col for col in df.columns 
                              if col.endswith(f'_{current_month}') and col not in id_cols
                              and not col.startswith(('fall_count_', 'fall_alarm_count_', 'brand_'))]
                
                if not feature_cols:
                    continue
                
                X_month = df[feature_cols].copy()
                X_month.columns = [col.replace(f'_{current_month}', '') for col in X_month.columns]
                X_month['age'] = df['age'].values if 'age' in df.columns else df.get('Age', 75)
                X_month['historical_data_quality'] = df[f'historical_data_quality_{i}'].values
                
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
        
        if 'age' in X.columns:
            X['age'] = X['age'].fillna(75)
        
        X = self._add_engineered_features(X)
        return X, y, account_numbers, current_month if predict_month else "multiple_months"
    
    def _calculate_steps_temporal(self, df, month_order, current_month):
        """Calculate Steps statistics using only data UP TO current_month"""
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
        
        # Coerce to numeric so empty/None become NaN (safe for np.nanmean and XGBoost missing)
        steps_data = df[steps_cols].apply(pd.to_numeric, errors='coerce').values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df['Steps_mean_temporal'] = np.nanmean(steps_data, axis=1)
            df['Steps_median_temporal'] = np.nanmedian(steps_data, axis=1)
            df['Steps_divergence_temporal'] = np.nanstd(steps_data, axis=1)
            df['Steps_Max_temporal'] = np.nanmax(steps_data, axis=1)
        return df
    
    def _add_engineered_features(self, X):
        """Add engineered features (NaN-aware)"""
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
        """Calculate optimal threshold to achieve target recall"""
        if len(np.unique(y)) < 2:
            print(f"    WARNING: Only one class, using default threshold")
            return self.optimal_threshold
        
        y_proba = self.predict_proba(X)
        p_min, p_max = np.percentile(y_proba, [0.1, 95])
        thresholds = np.linspace(max(p_min, 0.001), min(p_max, 0.5), 300)
        
        print(f"    Searching {len(thresholds)} thresholds, target recall >= {target_recall:.1%}")
        
        valid_thresholds = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            if y_pred.sum() == 0:
                continue
            
            recall = recall_score(y, y_pred, zero_division=0)
            precision = precision_score(y, y_pred, zero_division=0)
            
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

            fpr = fp / (fp + tn)
            fnr = fn / (fn + tp)


            
            
            if recall >= target_recall:
                valid_thresholds.append({
                    'threshold': threshold,
                    'recall': recall,
                    'precision': precision,
                    'f2': fbeta_score(y, y_pred, beta=2.0, zero_division=0),
                    'fpr': fpr,
                    'fnr': fnr
                })
        
        if not valid_thresholds:
            print(f"    WARNING: No threshold achieves {target_recall:.1%} recall, finding maximum recall")
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
            print(f"    Found {len(valid_thresholds)} thresholds meeting target")
            valid_thresholds.sort(key=lambda x: x['precision'], reverse=True)
            best_threshold = valid_thresholds[0]['threshold']
        
        y_pred_best = (y_proba >= best_threshold).astype(int)
        final_precision = precision_score(y, y_pred_best, zero_division=0)
        final_recall = recall_score(y, y_pred_best, zero_division=0)
        final_f2 = fbeta_score(y, y_pred_best, beta=2.0, zero_division=0)
        
        print(f"    Optimal: {best_threshold:.4f} (Recall={final_recall:.4f}, Precision={final_precision:.4f}, F2={final_f2:.4f}, FPR = {fpr:.4f}, FNR = {fnr:.4f})")
        
        return best_threshold
    
    def train(self, X_train, y_train, X_val=None, y_val=None, sample_weights=None, predict_month=None):
        """Train model with proper validation"""
        self.feature_names = list(X_train.columns)
        
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        if len(class_weights) == 2:
            class_weights[1] *= 9.5
        
        print(f"    Class distribution: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"    Class weights: {dict(zip(classes, class_weights))}")
        
        if sample_weights is None:
            quality_col = 'historical_data_quality' if 'historical_data_quality' in X_train.columns else 'data_quality_score'
            if quality_col in X_train.columns:
                completeness = X_train[quality_col].values
            else:
                feature_cols = [col for col in X_train.columns if col not in ['age', 'data_quality_score', 'historical_data_quality']]
                completeness = X_train[feature_cols].notna().sum(axis=1) / len(feature_cols)
            sample_weights = 1.0 + completeness
            print(f"    Sample weights: min={sample_weights.min():.2f}, max={sample_weights.max():.2f}, mean={sample_weights.mean():.2f}")
        
        if not self.use_smote:
            print(f"    SMOTE disabled (prevents temporal mixing)")
        
        # Coerce object/string columns to numeric so empty/None become NaN (XGBoost-compatible)
        obj_cols = X_train.select_dtypes(include=['object', 'str']).columns
        for col in obj_cols:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        if X_val is not None:
            for col in X_val.select_dtypes(include=['object', 'str']).columns:
                X_val[col] = pd.to_numeric(X_val[col], errors='coerce')
        
        X_train_scaled = self.scaler.fit_transform(X_train) if self.use_scaling else X_train
        if self.use_scaling:
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        if X_val is not None and y_val is not None and self.algorithm == 'XGBoost':
            X_val_scaled = self.scaler.transform(X_val) if self.use_scaling else X_val
            if self.use_scaling:
                X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
        
        self.model = self._create_model(class_weights if self.algorithm == 'XGBoost' else None)
        
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
        
        self.calibrated_model = None
        print(f"    Calibration: DISABLED (using native XGBoost probabilities)")
        
        if X_val is not None and y_val is not None:
            print(f"    Calculating optimal threshold...")
            optimal_threshold = self._calculate_optimal_threshold(X_val, y_val, method='recall_target', target_recall=0.70)
            
            if predict_month:
                self.adaptive_thresholds[predict_month] = optimal_threshold
                print(f"    Stored adaptive threshold for {predict_month}: {optimal_threshold:.4f}")
            else:
                self.optimal_threshold = optimal_threshold
        
        print(f"  Model trained: {self.algorithm}")
    def predict_proba(self, X):
        """Predict probabilities"""
        X_aligned = X[self.feature_names].copy()
        for col in X_aligned.select_dtypes(include=['object', 'str']).columns:
            X_aligned[col] = pd.to_numeric(X_aligned[col], errors='coerce')
        if self.use_scaling:
            X_aligned = X_aligned.fillna(0)
        X_scaled = self.scaler.transform(X_aligned) if self.use_scaling else X_aligned
        if self.use_scaling:
            X_scaled = pd.DataFrame(X_scaled, columns=X_aligned.columns)
        proba = self.calibrated_model.predict_proba(X_scaled) if self.calibrated_model else self.model.predict_proba(X_scaled)
        return proba[:, 1]
    
    def get_adaptive_threshold(self, predict_month):
        """Get adaptive threshold based on prediction month"""
        latest_month = max(self.adaptive_thresholds.keys(), key=lambda m: datetime.strptime(m, '%m_%Y')) if predict_month not in self.adaptive_thresholds else predict_month
        latest_optimal_threshold = self.adaptive_thresholds.get(latest_month,self.optimal_threshold)
        threshold = self.adaptive_thresholds.get(predict_month, latest_optimal_threshold)
        threshold_type = "recall-optimized" if predict_month in self.adaptive_thresholds else "recall-optimized fallback" if latest_month in self.adaptive_thresholds else "default"
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
    return 'High' if risk_score >= 7 else 'Moderate' if risk_score >= 3 else 'Low'


def plot_roc_auc(y_true, y_proba, title, output_file):
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


def _build_incremental_training_df(train_df, test_df, target_month):
    """Build next incremental training dataframe:
    - add new members from test
    - remove canceled members not present in test
    - append/update target_month columns from test
    """
    # Normalize join key type/value to avoid str vs int merge failures
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['account_number'] = train_df['account_number'].astype(str).str.strip()
    test_df['account_number'] = test_df['account_number'].astype(str).str.strip()
    
    id_cols = [c for c in ['account_number', 'account_id', 'health_plan', 'age', 'Age', 'brand']
               if c in train_df.columns or c in test_df.columns]
    new_month_cols = [c for c in test_df.columns if c not in id_cols and target_month in c]
    test_id_cols = [c for c in id_cols if c in test_df.columns]
    test_merge_cols = [c for c in test_id_cols + new_month_cols if c in test_df.columns]
    test_merge_cols = list(dict.fromkeys(test_merge_cols))
    test_df = test_df.drop_duplicates(subset=['account_number'], keep='first').copy()
    
    existing_new = [c for c in new_month_cols if c in train_df.columns]
    if existing_new:
        train_df = train_df.drop(columns=existing_new)
    
    merged_df = train_df.merge(test_df[test_merge_cols], on='account_number', how='outer')
    merged_df = merged_df[[c for c in merged_df.columns if not c.endswith('_duplicate')]]
    
    for col in test_id_cols:
        if col == 'account_number':
            continue
        x_col, y_col = f"{col}_x", f"{col}_y"
        if x_col in merged_df.columns and y_col in merged_df.columns:
            merged_df[col] = merged_df[x_col].fillna(merged_df[y_col])
            merged_df = merged_df.drop(columns=[x_col, y_col])

    # Ensure brand stays as ID column (not monthly feature columns like brand_MM_YYYY)
    brand_month_cols = [c for c in merged_df.columns if re.match(r'^brand_\d{2}_\d{4}$', c)]
    if brand_month_cols:
        month_brand = merged_df[brand_month_cols].bfill(axis=1).iloc[:, 0]
        if 'brand' in merged_df.columns:
            merged_df['brand'] = merged_df['brand'].fillna(month_brand)
        else:
            merged_df['brand'] = month_brand
        merged_df = merged_df.drop(columns=brand_month_cols)
    
    # Remove canceled members: keep only account_numbers present in latest real monthly file
    test_accounts = set(test_df['account_number'].astype(str))
    merged_df = merged_df[merged_df['account_number'].astype(str).isin(test_accounts)].copy()
    
    suffix_cols = [c for c in merged_df.columns if re.search(r'\.\d+$', c)]
    for col in suffix_cols:
        base = re.sub(r'\.\d+$', '', col)
        merged_df = merged_df.drop(columns=[col]) if base in merged_df.columns else merged_df.rename(columns={col: base})
    
    steps_cols = [c for c in merged_df.columns if c.startswith('Steps_')]
    merged_df = merged_df.drop(columns=steps_cols)
    
    id_order = ['account_number', 'account_id', 'health_plan', 'age', 'Age', 'brand']
    id_cols_first = [c for c in id_order if c in merged_df.columns]
    feature_cols = [c for c in merged_df.columns if c not in id_cols_first]
    merged_df = merged_df[id_cols_first + feature_cols]
    merged_df = merged_df.sort_values(
        ['health_plan', 'account_number'] if 'health_plan' in merged_df.columns else ['account_number']
    ).reset_index(drop=True)
    return merged_df


def update_training_file(old_file, test_file, new_file, target_month):
    print(f"\n  Updating training file with {target_month} data...")
    train_df = read_dataframe(old_file)
    test_df = read_dataframe(test_file)
    merged_df = _build_incremental_training_df(train_df, test_df, target_month)
    save_dataframe(merged_df, new_file)
    print(f"    New file: {new_file}, Shape: {merged_df.shape}")


# ============================================================================
# AUTOMATION HELPERS
# ============================================================================

def find_latest_training_file():
    """Find the most recent training file"""
    files = glob.glob("FallRisk_Training_*_To_*_Healthplans.*")
    
    if not files:
        raise FileNotFoundError("No training files found")
    
    file_dates = []
    for file in files:
        match = re.search(r'_To_(\d{2})(\d{4})_Healthplans', file)
        if match:
            mm, yyyy = int(match.group(1)), int(match.group(2))
            file_dates.append((yyyy * 100 + mm, file))
    
    if not file_dates:
        raise ValueError("Cannot parse dates from training filenames")
    
    file_dates.sort(reverse=True)
    latest_file = file_dates[0][1]
    
    print(f"[OK] Auto-detected: {latest_file}")
    return latest_file


def extract_start_month_from_file(filename):
    """Extract starting month (MMYYYY format)"""
    match = re.search(r'FallRisk_Training_(\d{6})_To_', filename)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot parse starting month from: {filename}")


def extract_last_month_from_file(filename):
    """Extract last month (MM_YYYY format)"""
    match = re.search(r'_To_(\d{2})(\d{4})_Healthplans', filename)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    raise ValueError(f"Cannot parse month from: {filename}")


def calculate_next_month(current_month):
    """Calculate next month (MM_YYYY format)"""
    mm, yyyy = current_month.split('_')
    mm_int, yyyy_int = int(mm), int(yyyy)
    
    if mm_int == 12:
        return f"01_{yyyy_int + 1}"
    return f"{mm_int + 1:02d}_{yyyy_int}"


def check_test_file_exists(month):
    """Check if test file exists for month"""
    pattern = f"FallRisk_Test_{month}_Healthplans"
    
    for ext in ['.parquet', '.csv', '.CSV']:
        file_path = pattern + ext
        if os.path.exists(file_path):
            return file_path
    return None


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def process_single_month(predict_month, training_file, test_file, algorithm):
    """Process single month: predict, evaluate, update"""
    print(f"\n{'='*80}\nPROCESSING: {predict_month}\n{'='*80}")
    
    mm, yyyy = predict_month.split('_')
    mm_yyyy = f"{mm}{yyyy}"
    
    prediction_dir = f"FallRisk_BetaVersion_Predict_{mm_yyyy}"
    testing_dir = f"FallRisk_BetaVersion_Test_Including_{mm_yyyy}"
    os.makedirs(prediction_dir, exist_ok=True)
    
    print(f"\n[A] Training and Predicting...")
    
    # Detect input file format
    is_parquet_input = training_file.lower().endswith('.parquet')
    output_format = 'parquet' if is_parquet_input else 'csv'
    print(f"  Input format: {output_format.upper()}")
    
    try:
        train_df = read_dataframe(training_file)
        
        # Stratified split (QA Fix #1)
        all_patient_ids = train_df['account_number'].unique()
        fall_count_cols = [c for c in train_df.columns if 'fall_count_' in c]
        patient_fall_status = train_df.groupby('account_number').apply(
            lambda x: (x[fall_count_cols].sum().sum() > 0) if fall_count_cols else False
        )
        
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(splitter.split(all_patient_ids, patient_fall_status[all_patient_ids]))
        train_patient_ids = all_patient_ids[train_idx]
        val_patient_ids = all_patient_ids[val_idx]
        
        train_fall_rate = patient_fall_status[train_patient_ids].mean()
        val_fall_rate = patient_fall_status[val_patient_ids].mean()
        print(f"  Patient split: {len(train_patient_ids)} train, {len(val_patient_ids)} val (NO overlap)")
        print(f"  Stratification: Train={train_fall_rate:.2%}, Val={val_fall_rate:.2%}")
        
        model = FallRiskForecastingModel(algorithm=algorithm, use_temporal_features=True, use_smote=False)
        
        X_train, y_train, _, _ = model.prepare_forecasting_features(train_df, predict_month=None, patient_ids_filter=train_patient_ids)
        X_val, y_val, _, _ = model.prepare_forecasting_features(train_df, predict_month=None, patient_ids_filter=val_patient_ids)
        print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features, Fall rate: {y_train.mean():.2%}")
        print(f"  Validation: {X_val.shape[0]} samples, Fall rate: {y_val.mean():.2%}")
        
        # Minimum validation check (QA Fix #3)
        val_positive_count = y_val.sum()
        if val_positive_count < 30:
            print(f"  WARNING: Only {val_positive_count} positive samples in validation")
        else:
            print(f"  [OK] Validation positive samples: {val_positive_count}")
        
        model.train(X_train, y_train, X_val=X_val, y_val=y_val, predict_month=predict_month)
        
        y_val_proba = model.predict_proba(X_val)
        y_val_pred = model.predict(X_val)
        val_roc_auc = roc_auc_score(y_val, y_val_proba) if len(np.unique(y_val)) > 1 else 0.0
        print(f"  Validation ROC-AUC: {val_roc_auc:.4f}")
        
        plot_roc_auc(y_val, y_val_proba, f'ROC-AUC Validation - {predict_month}',
                    os.path.join(prediction_dir, f'ROC_AUC_Val.png'))
        plot_confusion_matrix(y_val, y_val_pred, f'CM Validation - {predict_month}',
                            os.path.join(prediction_dir, f'CM_Val.png'))
        save_confusion_matrix_txt(y_val, y_val_pred, y_val_proba,
                                os.path.join(prediction_dir, f'CM_Val.txt'), f"Validation {predict_month}")
        evaluate_model(y_val, y_val_pred, y_val_proba, os.path.join(prediction_dir, f'Performance_Val.txt'))
        model.save_feature_importance(os.path.join(prediction_dir, f'FeatureImportance.txt'), 20)
        
        # Predict current month from current training roster only.
        # Member add/remove is applied later in [C] update_training_file for next month's training file.
        X_predict, _, account_numbers, _ = model.prepare_forecasting_features(train_df, predict_month=predict_month, patient_ids_filter=None)
        probabilities = model.predict_proba(X_predict)
        prob_dict = dict(zip(account_numbers, probabilities))
        data_quality_dict = dict(zip(account_numbers, X_predict['data_quality_score'].values))
        
        adaptive_threshold = model.get_adaptive_threshold(predict_month)
        
        forecast_data = []
        for i, row in train_df.iterrows():
            acc = row['account_number']
            prob = prob_dict.get(acc, 0.0)
            data_quality = data_quality_dict.get(acc, 0.0)
            
            forecast_data.append({
                'account_number': acc,
                'account_id': row.get('account_id', ''),
                'Age': row.get('Age', row.get('age', '')),
                'brand': row.get('brand', ''),
                'health_plan': row.get('health_plan', ''),
                'member name': row.get('member name', ''),
                'care manager': row.get('care manager', ''),
                'Probability': round(prob, 4),
                'Probability_Raw': prob,
                'Data_Quality_Pct': round(data_quality * 100, 1),
                'Flagged': prob >= adaptive_threshold
            })
        
        forecast_df = pd.DataFrame(forecast_data)
        
        def assign_risk_score_relative(prob, threshold):
            if prob >= threshold * 5.0: return 10
            elif prob >= threshold * 3.5: return 9
            elif prob >= threshold * 2.5: return 8
            elif prob >= threshold * 1.75: return 7
            elif prob >= threshold * 1.25: return 6
            elif prob >= threshold * 1.0: return 5
            elif prob >= threshold * 0.75: return 4
            elif prob >= threshold * 0.5: return 3
            elif prob >= threshold * 0.25: return 2
            else: return 1
        
        forecast_df["Risk_Score"] = forecast_df["Probability_Raw"].apply(
            lambda p: assign_risk_score_relative(p, adaptive_threshold)
        )
        forecast_df["Risk_Category"] = forecast_df["Risk_Score"].apply(get_risk_category)
        
        report_df = forecast_df.drop(columns=['Probability', 'Probability_Raw', 'Flagged'], errors='ignore')
        if is_parquet_input and HAS_PARQUET:
            forecast_file = os.path.join(prediction_dir, f'Fall_Risk_score_{mm_yyyy}.parquet')
            save_dataframe(report_df, forecast_file)
            print(f"  [OK] Forecast Parquet: {forecast_file}")
        else:
            forecast_file = os.path.join(prediction_dir, f'Fall_Risk_score_{mm_yyyy}.csv')
            save_dataframe(report_df, forecast_file)
            print(f"  [OK] Forecast CSV: {forecast_file}")
        
        print(f"  [OK] Total: {len(forecast_df)} members ({forecast_df['Flagged'].sum()} flagged)")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, False
    
    updated_training_file = training_file
    
    if test_file and os.path.exists(test_file):
        print(f"\n[B] Testing against real data...")
        try:
            os.makedirs(testing_dir, exist_ok=True)
            test_df = read_dataframe(test_file)
            test_df = test_df.copy()
            test_df['account_number'] = test_df['account_number'].astype(str).str.strip()
            
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
                forecast_indexed = forecast_df.copy()
                forecast_indexed['account_number'] = forecast_indexed['account_number'].astype(str).str.strip()
                forecast_indexed = forecast_indexed.set_index('account_number')
                matched = [a for a in test_df['account_number'].values if a in forecast_indexed.index]
                
                if matched:
                    y_pred_proba = np.array([forecast_indexed.loc[a, 'Probability_Raw'] for a in matched])
                    y_pred = np.array([int(forecast_indexed.loc[a, 'Flagged']) for a in matched])
                    y_true = np.array([y_actual[test_df['account_number'] == a].iloc[0] for a in matched])
                    
                    print(f"  Predicted: {y_pred.sum()}, Actual: {y_true.sum()}")
                    
                    if len(np.unique(y_true)) > 1:
                        print(f"  Test ROC-AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")
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
        
        print(f"\n[C] Updating training file...")
        try:
            start_month = extract_start_month_from_file(training_file)
            
            # Generate new training file in same format as input
            if is_parquet_input and HAS_PARQUET:
                # Parquet input -> Parquet output only
                new_training_file = f"FallRisk_Training_{start_month}_To_{mm_yyyy}_Healthplans.parquet"
                update_training_file(training_file, test_file, new_training_file, predict_month)
                print(f"  [OK] Training Parquet: {new_training_file}")
                updated_training_file = new_training_file
            else:
                # CSV input -> CSV output only
                new_training_file = f"FallRisk_Training_{start_month}_To_{mm_yyyy}_Healthplans.csv"
                update_training_file(training_file, test_file, new_training_file, predict_month)
                print(f"  [OK] Training CSV: {new_training_file}")
                updated_training_file = new_training_file
            
            print(f"  [OK] Ready for next month")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None, False
    
    print(f"\n{'='*80}\nCOMPLETED: {predict_month}\n{'='*80}")
    return updated_training_file, True


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Fall Risk Forecasting - Incremental Mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python FallRisk_Healthplans_Manual.py --predict_month 05_2025 --training_file FallRisk_Training_022024_To_042025_Healthplans
  python FallRisk_Healthplans_Manual.py --auto
        """
    )
    
    parser.add_argument('--auto', action='store_true', help='Automated mode')
    parser.add_argument('--predict_month', type=str, default=None, help='Month (MM_YYYY)')
    parser.add_argument('--training_file', type=str, default=None, help='Training file path')
    parser.add_argument('--test_file', type=str, default=None, help='Test file path (optional)')
    parser.add_argument('--algorithm', type=str, choices=['XGBoost', 'RandomForest', 'GradientBoosting', 'LogisticRegression'],
                       default='XGBoost', help='Algorithm (default: XGBoost)')
    
    args = parser.parse_args()
    
    if not args.auto:
        if not args.predict_month:
            parser.error("--predict_month required for manual mode")
        if not args.training_file:
            parser.error("--training_file required for manual mode")
        if not re.match(r'^\d{2}_\d{4}$', args.predict_month):
            parser.error(f"Invalid month format: {args.predict_month}")
    
    return args


def main():
    """Incremental fall risk forecasting workflow"""
    
    args = parse_arguments()
    
    print("="*80 + "\nFALL RISK FORECASTING - INCREMENTAL MODE\n" + "="*80)
    print(f"Algorithm: {args.algorithm}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.auto:
        print(f"Mode: AUTOMATED\n" + "="*80)
        
        iteration = 1
        max_iterations = 20
        training_file = None
        
        while iteration <= max_iterations:
            print(f"\n{'#'*80}\nITERATION {iteration}\n{'#'*80}\n")
            
            if training_file is None:
                try:
                    training_file = find_latest_training_file()
                except (FileNotFoundError, ValueError) as e:
                    print(f"\nERROR: {e}")
                    sys.exit(1)
            
            last_trained_month = extract_last_month_from_file(training_file)
            predict_month = calculate_next_month(last_trained_month)
            
            print(f"\nLast trained: {last_trained_month}")
            print(f"Predicting: {predict_month}")
            
            test_file = check_test_file_exists(predict_month)
            
            if test_file is None:
                print(f"\nNO TEST DATA FOR {predict_month}")
                print(f"Generating predictions only...\n")
                
                updated_file, success = process_single_month(
                    predict_month=predict_month,
                    training_file=training_file,
                    test_file=None,
                    algorithm=args.algorithm
                )
                
                if success:
                    print(f"\n{'='*80}")
                    print(f"[OK] PREDICTIONS GENERATED FOR {predict_month}")
                    print(f"{'='*80}")
                    print(f"Output: FallRisk_BetaVersion_Predict_{predict_month.replace('_', '')}/")
                    print(f"\nWaiting for: FallRisk_Test_{predict_month}_Healthplans")
                    print(f"Re-run with --auto when test file arrives")
                    print(f"{'='*80}\n")
                else:
                    print(f"\nERROR during prediction")
                    sys.exit(1)
                
                break
            
            print(f"\n[OK] Test file: {test_file}\n")
            
            updated_file, success = process_single_month(
                predict_month=predict_month,
                training_file=training_file,
                test_file=test_file,
                algorithm=args.algorithm
            )
            
            if not success:
                print(f"\nERROR processing {predict_month}")
                sys.exit(1)
            
            print(f"\n[OK] Processed {predict_month}")
            print(f"Training updated: {updated_file}")
            training_file = updated_file
            print(f"\nMoving to next month...\n")
            
            iteration += 1
        
        if iteration > max_iterations:
            print(f"\nReached max iterations ({max_iterations})")
        
        print(f"\n{'='*80}")
        print(f"AUTOMATED WORKFLOW COMPLETE")
        print(f"{'='*80}")
        print(f"Processed {iteration - 1} month(s)")
        print(f"{'='*80}\n")
    
    else:
        print(f"Mode: MANUAL")
        print(f"Month: {args.predict_month}")
        print(f"Training: {args.training_file}")
        if args.test_file:
            print(f"Test: {args.test_file}")
        print("="*80 + "\n")
        
        training_file = detect_file_format(args.training_file)
        if not os.path.exists(training_file):
            print(f"\nERROR: Training file not found: {training_file}")
            sys.exit(1)
        
        print(f"[OK] Training file: {training_file}")
        print(f"Processing: {args.predict_month}\n")
        
        test_file = args.test_file
        if test_file:
            test_file = detect_file_format(test_file)
        else:
            auto_test = check_test_file_exists(args.predict_month)
            if auto_test:
                print(f"[OK] Auto-detected test: {auto_test}\n")
                test_file = auto_test
        
        updated_file, success = process_single_month(
            predict_month=args.predict_month,
            training_file=training_file,
            test_file=test_file,
            algorithm=args.algorithm
        )
        
        if not success:
            print(f"\nERROR processing {args.predict_month}")
            sys.exit(1)
        
        print(f"\n{'='*80}\nPROCESSING COMPLETE\n{'='*80}")
        print(f"Month: {args.predict_month}")
        print(f"Algorithm: {args.algorithm}")
        
        if updated_file and updated_file != training_file:
            print(f"Updated training: {updated_file}")
            print(f"\nNext month: use --training_file {updated_file}")
        else:
            print(f"Training: {training_file} (unchanged)")
            if not test_file:
                print(f"\nWhen actuals arrive:")
                print(f"Re-run with --test_file to evaluate")
        
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
