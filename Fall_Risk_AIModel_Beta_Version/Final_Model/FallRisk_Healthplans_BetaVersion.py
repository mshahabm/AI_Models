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
    
    def __init__(self, algorithm='XGBoost'):
        self.algorithm = algorithm
        self.model = None
        self.calibrated_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.use_scaling = algorithm == 'LogisticRegression'
        self.optimal_threshold = 0.22
        
    def _create_model(self, class_weights=None):
        """Create model instance"""
        if self.algorithm == 'XGBoost':
            scale_pos_weight = 12.0
            if class_weights is not None and len(class_weights) == 2:
                scale_pos_weight = (class_weights[1] / class_weights[0] if class_weights[0] > 0 else 1.0) * 3.5
            
            return xgb.XGBClassifier(
                n_estimators=800, max_depth=3, learning_rate=0.01, subsample=0.9,
                colsample_bytree=0.9, colsample_bylevel=0.9, scale_pos_weight=scale_pos_weight,
                min_child_weight=1, gamma=0.05, reg_alpha=0.1, reg_lambda=0.8,
                random_state=42, eval_metric='logloss', use_label_encoder=False,
                tree_method='hist', max_bin=512, min_split_loss=0.1
            )
        elif self.algorithm == 'RandomForest':
            return RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=3,
                                         min_samples_leaf=1, class_weight='balanced', 
                                         random_state=42, n_jobs=-1)
        elif self.algorithm == 'GradientBoosting':
            return GradientBoostingClassifier(n_estimators=200, max_depth=7, learning_rate=0.05,
                                             subsample=0.8, random_state=42)
        elif self.algorithm == 'LogisticRegression':
            return LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced',
                                     C=0.1, solver='liblinear')
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def prepare_forecasting_features(self, df, predict_month=None):
        """Prepare features for forecasting (Month N → Month N+1)"""
        month_order = ['Nov_2024', 'Dec_2024', 'Jan_2025', 'Feb_2025', 'Mar_2025', 'Apr_2025',
                      'May_2025', 'Jun_2025', 'Jul_2025', 'Aug_2025', 'Sep_2025', 'Oct_2025',
                      'Nov_2025', 'Dec_2025', 'Jan_2026']
        
        id_cols = [col for col in ['account_number', 'account_id', 'brand', 'health_plan', 'age', 'Age'] 
                   if col in df.columns]
        
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
            
            feature_cols = [col for col in df.columns 
                          if col.endswith(f'_{current_month}') and col not in id_cols and
                          not col.startswith('fall_count_') and not col.startswith('fall_alarm_count_')]
            
            X = df[feature_cols].copy()
            X.columns = [col.replace(f'_{current_month}', '') for col in X.columns]
            X['age'] = df['age'].values if 'age' in df.columns else df.get('Age', 75)
            
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
                
                feature_cols = [col for col in df.columns 
                              if col.endswith(f'_{current_month}') and col not in id_cols and
                              not col.startswith('fall_count_') and not col.startswith('fall_alarm_count_')]
                
                if not feature_cols:
                    continue
                
                X_month = df[feature_cols].copy()
                X_month.columns = [col.replace(f'_{current_month}', '') for col in X_month.columns]
                X_month['age'] = df['age'].values if 'age' in df.columns else df.get('Age', 75)
                
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
        
        X = X.fillna(0)
        X = self._add_engineered_features(X)
        return X, y, account_numbers, current_month_out
    
    def _add_engineered_features(self, X):
        """Add engineered features"""
        X = X.copy()
        
        if 'age' in X.columns:
            if 'avg_daily_steps' in X.columns:
                X['age_x_steps'] = X['age'] * X['avg_daily_steps']
                X['age_x_steps_ratio'] = X['age'] / (X['avg_daily_steps'] + 1)
            if 'fall_alarm_count' in X.columns:
                X['age_x_fall_alarm'] = X['age'] * X['fall_alarm_count']
            if 'assist_count' in X.columns:
                X['age_x_assist'] = X['age'] * X['assist_count']
        
        if 'avg_daily_steps' in X.columns:
            X['steps_squared'] = X['avg_daily_steps'] ** 2
            X['low_activity_flag'] = (X['avg_daily_steps'] < 1000).astype(int)
        
        fall_cols = [col for col in X.columns if 'fall_' in col or 'assist' in col or 'dispatch' in col]
        if fall_cols:
            X['total_fall_events'] = X[fall_cols].sum(axis=1)
        
        if 'er_dispatch_count' in X.columns:
            X['has_er_dispatch'] = (X['er_dispatch_count'] > 0).astype(int)
        if 'sentiment_negative_count' in X.columns:
            X['has_negative_sentiment'] = (X['sentiment_negative_count'] > 0).astype(int)
        if 'fall_alarm_count' in X.columns and 'assist_count' in X.columns:
            X['fall_to_assist_ratio'] = X['fall_alarm_count'] / (X['assist_count'] + 1)
        if 'button_press_count' in X.columns and 'fall_alarm_count' in X.columns:
            X['button_to_fall_ratio'] = X['button_press_count'] / (X['fall_alarm_count'] + 1)
        
        return X
    
    def train(self, X_train, y_train, sample_weights=None):
        """Train the model"""
        self.feature_names = list(X_train.columns)
        
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        if len(class_weights) == 2:
            class_weights[1] *= 4.0
        
        print(f"    Class distribution: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"    Class weights: {dict(zip(classes, class_weights))}")
        
        # Apply SMOTE
        if HAS_SMOTE and len(classes) == 2:
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
        
        X_train_scaled = self.scaler.fit_transform(X_train) if self.use_scaling else X_train
        if self.use_scaling:
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        self.model = self._create_model(class_weights if self.algorithm == 'XGBoost' else None)
        self.model.fit(X_train_scaled, y_train, sample_weight=sample_weights) if sample_weights else self.model.fit(X_train_scaled, y_train)
        
        # Calibrate
        try:
            print(f"    Calibrating probabilities...")
            self.calibrated_model = CalibratedClassifierCV(self.model, method='isotonic', cv=5, ensemble=True)
            self.calibrated_model.fit(X_train_scaled, y_train)
        except Exception as e:
            print(f"    Calibration failed: {e}")
            self.calibrated_model = None
        
        print(f"  Model trained: {self.algorithm}")
    
    def predict_proba(self, X):
        """Predict probabilities"""
        X_aligned = X[self.feature_names].copy().fillna(0)
        X_scaled = self.scaler.transform(X_aligned) if self.use_scaling else X_aligned
        if self.use_scaling:
            X_scaled = pd.DataFrame(X_scaled, columns=X_aligned.columns)
        
        proba = self.calibrated_model.predict_proba(X_scaled) if self.calibrated_model else self.model.predict_proba(X_scaled)
        return proba[:, 1]
    
    def predict(self, X, threshold=None):
        """Predict classes"""
        threshold = threshold or self.optimal_threshold
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
                        'optimal_threshold': self.optimal_threshold}, f)
    
    @classmethod
    def load(cls, filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        instance = cls(data['algorithm'])
        for key in ['model', 'calibrated_model', 'scaler', 'feature_names', 'use_scaling', 'optimal_threshold']:
            setattr(instance, key, data.get(key, getattr(instance, key)))
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


def calculate_steps_parameters(df, month_order):
    """Calculate Steps statistics"""
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
                   'Nov_2025', 'Dec_2025', 'Jan_2026']
    
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
    
    # Recalculate Steps
    print(f"\n    Recalculating Steps parameters...")
    steps_params = ['Steps_mean', 'Steps_median', 'Steps_divergence', 'Steps_Max']
    merged_df = merged_df.drop(columns=[c for c in steps_params if c in merged_df.columns])
    merged_df = calculate_steps_parameters(merged_df, month_order)
    
    merged_df = merged_df.sort_values(['health_plan', 'account_number']).reset_index(drop=True)
    save_dataframe(merged_df, new_training_file)
    print(f"    New file: {new_training_file}, Shape: {merged_df.shape}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Automated 25-step forecasting workflow"""
    print("="*80 + "\nAUTOMATED FALL RISK FORECASTING\n" + "="*80)
    
    config = {
        'algorithm': 'XGBoost',
        'initial_training_file': 'FallRisk_Training_112024_To_042025_Healthplans.CSV',
        'output_base_dir': 'FallRisk_Model7',
        'months_to_process': [
            {'predict': 'May_2025', 'test_file': 'FallRisk_Test_May2025_Healthplans.CSV'},
            {'predict': 'Jun_2025', 'test_file': 'FallRisk_Test_June2025_Healthplans.CSV'},
            {'predict': 'Jul_2025', 'test_file': 'FallRisk_Test_July2025_Healthplans.CSV'},
            {'predict': 'Aug_2025', 'test_file': 'FallRisk_Test_August2025_Healthplans.CSV'},
            {'predict': 'Sep_2025', 'test_file': 'FallRisk_Test_September2025_Healthplans.CSV'},
            {'predict': 'Oct_2025', 'test_file': 'FallRisk_Test_October2025_Healthplans.csv'},
            {'predict': 'Nov_2025', 'test_file': 'FallRisk_Test_November2025_Healthplans.csv'},
            {'predict': 'Dec_2025', 'test_file': 'FallRisk_Test_December2025_Healthplans.csv'},
            {'predict': 'Jan_2026', 'test_file': None}
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
        prediction_dir = f"{config['output_base_dir']}_Predicting{month_short}{year_short}"
        testing_dir = f"{config['output_base_dir']}_Test_Against{month_short}{year_short}"
        os.makedirs(prediction_dir, exist_ok=True)
        
        # STEP A: Train & Predict
        print(f"\n[{idx}A] Training and Predicting...")
        try:
            train_df = read_dataframe(current_training_file)
            model = FallRiskForecastingModel(algorithm=config['algorithm'])
            
            X_train, y_train, _, _ = model.prepare_forecasting_features(train_df, predict_month=None)
            print(f"  Samples: {X_train.shape[0]}, Features: {X_train.shape[1]}, Fall rate: {y_train.mean():.2%}")
            
            model.train(X_train, y_train)
            
            # Validation
            y_val_proba = model.predict_proba(X_train)
            y_val_pred = model.predict(X_train)
            print(f"  Validation ROC-AUC: {roc_auc_score(y_train, y_val_proba):.4f}")
            
            plot_roc_auc(y_train, y_val_proba, f'ROC-AUC Validation - {predict_month}',
                        os.path.join(prediction_dir, f'ROC_AUC_Val_{month_short}.png'))
            plot_confusion_matrix(y_train, y_val_pred, f'Confusion Matrix Validation - {predict_month}',
                                os.path.join(prediction_dir, f'CM_Val_{month_short}.png'))
            save_confusion_matrix_txt(y_train, y_val_pred, y_val_proba,
                                    os.path.join(prediction_dir, f'CM_Val_{month_short}.txt'),
                                    f"Validation {predict_month}")
            evaluate_model(y_train, y_val_pred, y_val_proba,
                          os.path.join(prediction_dir, f'Performance_Val_{month_short}.txt'))
            model.save_feature_importance(os.path.join(prediction_dir, f'FeatureImportance_{month_short}.txt'), 20)
            
            # Generate predictions
            X_predict, _, account_numbers, _ = model.prepare_forecasting_features(train_df, predict_month=predict_month)
            probabilities = model.predict_proba(X_predict)
            
            forecast_data = []
            for i, row in train_df.iterrows():
                acc = row['account_number']
                prob = probabilities[list(account_numbers).index(acc)] if acc in account_numbers else 0.0
                forecast_data.append({
                    'account_number': acc,
                    'account_id': row.get('account_id', ''),
                    'Age': row.get('Age', row.get('age', '')),
                    'brand': row.get('brand', ''),
                    'health_plan': row.get('health_plan', ''),
                    'member name': row.get('member name', ''),
                    'care manager': row.get('care manager', ''),
                    'Risk_Score': calculate_risk_score(prob),
                    'Risk_Category': get_risk_category(calculate_risk_score(prob))
                })
            
            forecast_df = pd.DataFrame(forecast_data)
            file_ext = '.parquet' if current_training_file.lower().endswith('.parquet') else '.CSV'
            forecast_file = os.path.join(prediction_dir, f'FallRiskForecast_{month_short}_{year_short}_Healthplans{file_ext}')
            save_dataframe(forecast_df, forecast_file)
            print(f"  ✓ Forecast: {len(forecast_df)} members, {forecast_df['Risk_Category'].value_counts().to_dict()}")
            
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
                        y_pred_proba = [(forecast_indexed.loc[a, 'Risk_Score'] - 1) / 9.0 for a in matched]
                        y_pred = [1 if p >= model.optimal_threshold else 0 for p in y_pred_proba]
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
