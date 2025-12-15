"""
Fall Risk Prediction Model - Model 3
Predicts fall risk after 1, 2, and 3 months using XGBoost

Requirements:
- Load data from active_dtc_w_llm_features.parquet
- Create targets for 1, 2, and 3 months ahead
- Feature engineering with lag and rolling features
- 70/30 train/test split
- Train XGBoost models for each horizon
- Generate ROC-AUC curves (training and testing)
- Generate confusion matrices (training and testing)
- Generate reports (model performance, metrics summary, all features)
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = 'Fall_alarm_llm_count_target_column.parquet'
HORIZONS = [1, 2, 3]  # 1, 2, and 3 months ahead
TARGET_COLUMN = 'target'  # This is now the fall count (0-20) from LLM detection
TRAIN_TEST_SPLIT = 0.5  # 50% training, 50% testing
TRACKING_COLUMNS = ['account_number', 'brand']

# XGBoost parameters - Optimized for better precision
XGBOOST_PARAMS = {
    'n_estimators': 150,
    'learning_rate': 0.03,
    'max_depth': 3,  # Reduced to prevent overfitting and false positives
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,  # Increased regularization to reduce false positives
    'reg_lambda': 3.0,  # Increased regularization
    'min_child_weight': 8,  # Increased to require more samples for splits
    'gamma': 0.2,  # Increased to reduce overfitting
    'random_state': 42,
    'eval_metric': 'auc',
    'n_jobs': -1,
    'tree_method': 'hist'
}

# Prediction threshold for better precision (higher threshold = fewer false positives)
PREDICTION_THRESHOLD = 0.5  # Will be optimized based on precision-recall tradeoff

# Output directories
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set style for plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")


def load_data():
    """Load data from parquet file"""
    print(f"\n{'='*70}")
    print("Loading Data")
    print(f"{'='*70}")
    
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    
    df = pd.read_parquet(DATA_FILE)
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    # Remove obs_month column as it has inaccurate data
    if 'obs_month' in df.columns:
        print(f"\nRemoving 'obs_month' column (inaccurate data)...")
        df = df.drop(columns=['obs_month'])
        print(f"  ✅ Removed 'obs_month' column")
        print(f"  Remaining columns: {len(df.columns)}")
    
    # Basic data quality checks
    print(f"\nData Quality:")
    print(f"  - Missing values: {df.isnull().sum().sum()}")
    print(f"  - Unique accounts: {df['account_number'].nunique():,}")
    
    # Verify each account has 6 months of data (assuming data is accurate)
    account_counts = df.groupby('account_number').size()
    print(f"  - Accounts with 6 months of data: {(account_counts == 6).sum():,}")
    print(f"  - Accounts with < 6 months: {(account_counts < 6).sum():,}")
    print(f"  - Accounts with > 6 months: {(account_counts > 6).sum():,}")
    if len(account_counts) > 0:
        print(f"  - Average rows per account: {account_counts.mean():.2f}")
    
    if TARGET_COLUMN in df.columns:
        target_dist = df[TARGET_COLUMN].value_counts().to_dict()
        print(f"  - Target distribution: {target_dist}")
    
    print(f"\nNote: All rows are kept. Data is assumed to be accurate for last 6 months per account.")
    print(f"      Rows are processed in their current order (assumed chronological per account).")
    
    return df


def create_targets(df):
    """
    Create binary targets for 1, 2, and 3 month predictions.
    Target column contains actual fall count for each month (0-20).
    Creates binary targets: 1 if any fall (count > 0) occurs in future months, 0 otherwise.
    """
    print(f"\n{'='*70}")
    print("Creating Temporal Targets")
    print(f"{'='*70}")
    
    # Sort by account_number to ensure consistent ordering
    # Data structure: Each account has 6 rows (one per month, May-Oct 2025)
    # Rows are assumed to be in chronological order per account
    df = df.sort_values('account_number').reset_index(drop=True)
    
    print(f"Processing {df['account_number'].nunique():,} accounts...")
    print(f"  - Each account has 6 months of data (May-Oct 2025)")
    print(f"  - Target column contains actual fall count for that month (0-20)")
    print(f"  - Creating binary targets: 1 if fall occurs in future months, 0 otherwise")
    
    def create_targets_for_account(group):
        """Create targets for a single account (6 months of data)"""
        # Keep original order (assumed to be chronological: month 1, 2, 3, 4, 5, 6)
        n = len(group)
        target_vals = group[TARGET_COLUMN].values.astype(float)
        
        # Create targets for each horizon
        # Target is count-based (0-20), so we create binary targets: 1 if any fall (count > 0) in future, 0 otherwise
        for horizon in HORIZONS:
            target_col = f'target_{horizon}m'
            future_falls = np.full(n, np.nan, dtype=float)
            
            # For each row, check if there's a fall (count > 0) in the next 'horizon' months
            for i in range(n):
                # Look ahead 'horizon' months
                end_idx = min(i + horizon + 1, n)
                if i + 1 < end_idx:
                    # Check if any fall occurred in the future window (count > 0)
                    future_window = target_vals[i+1:end_idx]
                    # Binary target: 1 if any month has fall count > 0, 0 otherwise
                    future_falls[i] = 1.0 if np.any(future_window > 0) else 0.0
            
            group[target_col] = future_falls
        
        return group
    
    # Apply to each account
    print("Creating targets for each account...")
    df = df.groupby('account_number', group_keys=False).apply(create_targets_for_account)
    df = df.reset_index(drop=True)
    
    # Report target distributions
    for horizon in HORIZONS:
        target_col = f'target_{horizon}m'
        valid_targets = df[target_col].dropna()
        if len(valid_targets) > 0:
            dist = valid_targets.value_counts().to_dict()
            print(f"\n{horizon}-month target distribution:")
            print(f"  - Total valid: {len(valid_targets):,}")
            print(f"  - Distribution: {dist}")
            print(f"  - Positive class ratio: {valid_targets.mean():.4f}")
    
    return df


def create_features(df):
    """Create feature engineering: lag features and rolling features"""
    print(f"\n{'='*70}")
    print("Feature Engineering")
    print(f"{'='*70}")
    
    # Sort by account_number to ensure consistent ordering
    df = df.sort_values('account_number').reset_index(drop=True)
    
    # Features to create lag and rolling features for
    # Note: 'target' is excluded as it's the target variable, not a feature
    feature_cols = [
        'avg_daily_steps', 'prev_avg_daily_steps', 'steps_delta',
        'button_press_count', 'er_dispatch_count', 'fall_alarm_count',
        'sentiment_negative_count', 'sentiment_positive_count', 
        'sentiment_neutral_count', 'assist_lift_or_stand_flag_count',
        'fall_detect_enabled', 'age'
    ]
    
    # Only use features that exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    print(f"Creating features for: {feature_cols}")
    
    # Create lag features (1, 2, 3 months ago)
    print("\nCreating lag features...")
    lag_periods = [1, 2, 3]
    for feature in feature_cols:
        for lag in lag_periods:
            df[f'{feature}_lag{lag}'] = df.groupby('account_number')[feature].shift(lag)
    
    # Create rolling features (using consistent 1-month window - monthly rollover)
    print("Creating rolling features with consistent 1-month window (monthly rollover)...")
    rolling_window = 1  # Monthly rollover (1-month window)
    for feature in feature_cols:
        # Rolling mean
        df[f'{feature}_rolling_mean_{rolling_window}m'] = df.groupby('account_number')[feature].transform(
            lambda x: x.shift(1).rolling(window=rolling_window, min_periods=1).mean()
        )
        # Rolling std
        df[f'{feature}_rolling_std_{rolling_window}m'] = df.groupby('account_number')[feature].transform(
            lambda x: x.shift(1).rolling(window=rolling_window, min_periods=1).std()
        )
    
    # Create trend features (slope over last 3 months)
    print("Creating trend features...")
    for feature in feature_cols:
        def calculate_trend(group):
            values = group[feature].values
            trend = np.zeros(len(values))
            for i in range(3, len(values)):
                window = values[i-3:i]
                if len(window) > 1:
                    valid = ~np.isnan(window)
                    if valid.sum() > 1:
                        x = np.arange(len(window))[valid]
                        y = window[valid]
                        if len(x) > 1:
                            slope = np.polyfit(x, y, 1)[0]
                            trend[i] = slope
            return pd.Series(trend, index=group.index)
        
        df[f'{feature}_trend_3m'] = df.groupby('account_number', group_keys=False).apply(calculate_trend).reset_index(drop=True)
    
    # Skip time-based features (obs_month removed due to inaccurate data)
    print("Skipping time-based features (obs_month removed)...")
    
    # Fill NaN values
    print("Filling missing values...")
    # Fill lag features with 0
    lag_cols = [col for col in df.columns if '_lag' in col]
    df[lag_cols] = df[lag_cols].fillna(0)
    
    # Fill rolling features with forward fill then 0
    rolling_cols = [col for col in df.columns if 'rolling' in col]
    for col in rolling_cols:
        df[col] = df.groupby('account_number')[col].ffill().fillna(0)
    
    # Fill trend features with 0
    trend_cols = [col for col in df.columns if '_trend' in col]
    df[trend_cols] = df[trend_cols].fillna(0)
    
    # Count created features
    new_features = lag_cols + rolling_cols + trend_cols
    print(f"\nFeature Engineering Summary:")
    print(f"  - Lag features: {len(lag_cols)} (1, 2, 3 months)")
    print(f"  - Rolling features: {len(rolling_cols)} (consistent 1-month window - monthly rollover)")
    print(f"  - Trend features: {len(trend_cols)} (3-month trend)")
    print(f"  - Time features: 0 (obs_month removed)")
    print(f"  - Total new features: {len(new_features)}")
    print(f"  - Note: Rolling features use consistent 1-month window (monthly rollover)")
    
    return df


def prepare_features_for_model(df):
    """Prepare features excluding tracking and target columns"""
    # Exclude tracking columns and target columns
    exclude_cols = ['account_number', 'brand', TARGET_COLUMN] + [f'target_{h}m' for h in HORIZONS]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Convert object columns to category for XGBoost
    for col in feature_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
    
    return df, feature_cols


def train_test_split(df, split_ratio=TRAIN_TEST_SPLIT):
    """
    Perform simple 70/30 train/test split
    Split by accounts to ensure all rows for an account are in the same set
    """
    print(f"\n{'='*70}")
    print("Train/Test Split (50/50)")
    print(f"{'='*70}")
    
    # Get unique accounts
    unique_accounts = df['account_number'].unique()
    n_accounts = len(unique_accounts)
    
    # Shuffle accounts for random split
    np.random.seed(42)
    shuffled_accounts = np.random.permutation(unique_accounts)
    
    # Split accounts
    n_train_accounts = int(n_accounts * split_ratio)
    train_accounts = set(shuffled_accounts[:n_train_accounts])
    test_accounts = set(shuffled_accounts[n_train_accounts:])
    
    # Create masks
    train_mask = df['account_number'].isin(train_accounts)
    test_mask = df['account_number'].isin(test_accounts)
    
    print(f"Total accounts: {n_accounts:,}")
    print(f"Training accounts: {len(train_accounts):,} ({len(train_accounts)/n_accounts*100:.1f}%)")
    print(f"Testing accounts: {len(test_accounts):,} ({len(test_accounts)/n_accounts*100:.1f}%)")
    print(f"\nFinal Split:")
    print(f"Training samples: {train_mask.sum():,} ({train_mask.sum()/len(df)*100:.1f}%)")
    print(f"Testing samples: {test_mask.sum():,} ({test_mask.sum()/len(df)*100:.1f}%)")
    print(f"Accounts in training: {df[train_mask]['account_number'].nunique():,}")
    print(f"Accounts in testing: {df[test_mask]['account_number'].nunique():,}")
    print(f"Total unique accounts: {df['account_number'].nunique():,}")
    
    return train_mask, test_mask


def train_model(X_train, y_train, X_test, y_test, horizon, feature_cols):
    """Train XGBoost model for a specific horizon"""
    print(f"\n{'='*70}")
    print(f"Training {horizon}-Month Prediction Model")
    print(f"{'='*70}")
    
    # Handle class imbalance - Use more conservative weighting to reduce false positives
    params = XGBOOST_PARAMS.copy()
    if y_train.mean() < 0.5:
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        if pos_count > 0:
            # Use more conservative weight (reduced by 30%) to improve precision
            base_weight = neg_count / pos_count
            params['scale_pos_weight'] = base_weight * 0.7  # Reduce to improve precision
            print(f"Class imbalance detected. Using conservative scale_pos_weight: {params['scale_pos_weight']:.2f} (reduced from {base_weight:.2f})")
    
    # Initialize and train model
    model = XGBClassifier(**params)
    print(f"Training with {len(X_train):,} samples and {len(feature_cols)} features...")
    
    # Only use eval_set if test set is not empty
    if len(X_test) > 0 and len(y_test) > 0:
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
    else:
        print(f"⚠️  Warning: Test set is empty. Training without eval_set.")
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False
        )
    
    # Get probability predictions
    y_train_prob = model.predict_proba(X_train)[:, 1]
    
    # Handle empty test set
    if len(X_test) > 0:
        y_test_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_test_prob = np.array([])
    
    # Optimize threshold for better precision-recall balance
    # Find threshold that maximizes F1 score while maintaining reasonable precision
    print("\nOptimizing prediction threshold for better precision...")
    if len(y_train) > 0 and y_train.sum() > 0:
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_train, y_train_prob)
        
        # Note: precision_vals and recall_vals have one more element than thresholds
        # The last point (precision=0, recall=1) has no threshold, so we exclude it
        # Use only the first len(thresholds) elements
        precision_vals = precision_vals[:len(thresholds)]
        recall_vals = recall_vals[:len(thresholds)]
        
        # Find threshold that gives precision >= 0.3 (minimum acceptable) and maximizes F1
        f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
        
        # Prioritize precision: find threshold where precision >= 0.3 and F1 is maximized
        valid_indices = precision_vals >= 0.3
        if valid_indices.sum() > 0:
            # Get indices where precision >= 0.3
            valid_idx_array = np.where(valid_indices)[0]
            if len(valid_idx_array) > 0:
                best_idx_in_valid = np.argmax(f1_scores[valid_indices])
                best_idx = valid_idx_array[best_idx_in_valid]
                optimal_threshold = thresholds[best_idx]
                optimal_precision = precision_vals[best_idx]
                optimal_recall = recall_vals[best_idx]
                optimal_f1 = f1_scores[best_idx]
            else:
                # Fallback
                best_idx = np.argmax(precision_vals)
                optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
                optimal_precision = precision_vals[best_idx]
                optimal_recall = recall_vals[best_idx]
                optimal_f1 = f1_scores[best_idx]
        else:
            # If no threshold gives precision >= 0.3, use threshold that maximizes precision
            best_idx = np.argmax(precision_vals)
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            optimal_precision = precision_vals[best_idx]
            optimal_recall = recall_vals[best_idx]
            optimal_f1 = f1_scores[best_idx]
        
        print(f"  Optimal threshold: {optimal_threshold:.4f}")
        print(f"  Expected precision: {optimal_precision:.4f}, recall: {optimal_recall:.4f}, F1: {optimal_f1:.4f}")
    else:
        optimal_threshold = 0.5
        print(f"  Using default threshold: {optimal_threshold:.4f}")
    
    # Make predictions using optimized threshold
    y_train_pred = (y_train_prob >= optimal_threshold).astype(int)
    
    # Handle empty test set
    if len(X_test) > 0:
        y_test_pred = (y_test_prob >= optimal_threshold).astype(int)
    else:
        y_test_pred = np.array([])
    
    # Calculate metrics
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1': f1_score(y_train, y_train_pred, zero_division=0),
        'auc': roc_auc_score(y_train, y_train_prob),
        'threshold': optimal_threshold
    }
    
    # Handle empty test set
    if len(y_test) == 0:
        print(f"⚠️  Warning: Test set is empty. Setting test metrics to NaN.")
        test_metrics = {
            'accuracy': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'f1': np.nan,
            'auc': np.nan,
            'threshold': optimal_threshold
        }
    else:
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_test_prob),
            'threshold': optimal_threshold
        }
    
    print(f"\n✅ Model trained successfully!")
    print(f"Training - Accuracy: {train_metrics['accuracy']:.4f}, Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
    if len(y_test) > 0:
        print(f"Testing  - Accuracy: {test_metrics['accuracy']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")
    else:
        print(f"Testing  - No test samples available")
    
    # Save model
    model_file = os.path.join(MODELS_DIR, f'active_dtc_w_llm_features_{horizon}m_model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_cols': feature_cols,
            'horizon': horizon,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'optimal_threshold': optimal_threshold
        }, f)
    print(f"Model saved to: {model_file}")
    
    return {
        'model': model,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'y_train_prob': y_train_prob,
        'y_test_prob': y_test_prob,
        'feature_cols': feature_cols
    }


def plot_roc_curve(y_true, y_prob, horizon, dataset_type, save_dir, timestamp):
    """Plot ROC curve and save to file with unique timestamp"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {horizon}-Month Model ({dataset_type})', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(save_dir, f'roc_curve_{horizon}m_{dataset_type.lower()}_{timestamp}.png')
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ ROC curve saved: {filename}")


def plot_confusion_matrix(y_true, y_pred, horizon, dataset_type, save_dir, timestamp):
    """Plot confusion matrix and save to file with unique timestamp"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Calculate percentages
    total = cm.sum()
    percentages = (cm / total * 100).round(2) if total > 0 else np.zeros_like(cm)
    
    # Create annotations
    annot_array = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_array[i, j] = f'{cm[i, j]:,}\n({percentages[i, j]:.2f}%)'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=annot_array, fmt='', cmap='Blues', cbar=True,
                xticklabels=['No Fall', 'Fall'], yticklabels=['No Fall', 'Fall'],
                ax=ax, linewidths=1, linecolor='gray')
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {horizon}-Month Model ({dataset_type})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = os.path.join(save_dir, f'confusion_matrix_{horizon}m_{dataset_type.lower()}_{timestamp}.png')
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ Confusion matrix saved: {filename}")


def generate_reports(all_results, feature_cols, timestamp, df):
    """Generate all required reports with unique timestamps"""
    print(f"\n{'='*70}")
    print("Generating Reports")
    print(f"{'='*70}")
    
    # 1. Model Performance Report (with all horizons: 1m, 2m, 3m)
    print("\n1. Generating Model Performance Report...")
    performance_data = []
    for horizon in HORIZONS:
        if horizon in all_results:
            result = all_results[horizon]
            performance_data.append({
                'Horizon (months)': horizon,
                'Dataset': 'Training',
                'Accuracy': result['train_metrics']['accuracy'],
                'Precision': result['train_metrics']['precision'],
                'Recall': result['train_metrics']['recall'],
                'F1-Score': result['train_metrics']['f1'],
                'ROC-AUC': result['train_metrics']['auc']
            })
            performance_data.append({
                'Horizon (months)': horizon,
                'Dataset': 'Testing',
                'Accuracy': result['test_metrics']['accuracy'],
                'Precision': result['test_metrics']['precision'],
                'Recall': result['test_metrics']['recall'],
                'F1-Score': result['test_metrics']['f1'],
                'ROC-AUC': result['test_metrics']['auc']
            })
    
    performance_df = pd.DataFrame(performance_data)
    performance_file = os.path.join(RESULTS_DIR, f'model_performance_report_{timestamp}.csv')
    performance_df.to_csv(performance_file, index=False)
    print(f"  ✅ Saved: {performance_file}")
    print(f"  - Includes all horizons: 1m, 2m, 3m")
    print(f"  - Total rows: {len(performance_df)} (6 rows: 3 horizons × 2 datasets)")
    
    # 2. Metrics Summary Report
    print("\n2. Generating Metrics Summary Report...")
    summary_data = []
    for horizon in HORIZONS:
        if horizon in all_results:
            result = all_results[horizon]
            summary_data.append({
                'Horizon': f'{horizon} month(s)',
                'Train_Accuracy': result['train_metrics']['accuracy'],
                'Train_AUC': result['train_metrics']['auc'],
                'Test_Accuracy': result['test_metrics']['accuracy'],
                'Test_AUC': result['test_metrics']['auc'],
                'Train_Precision': result['train_metrics']['precision'],
                'Train_Recall': result['train_metrics']['recall'],
                'Train_F1': result['train_metrics']['f1'],
                'Test_Precision': result['test_metrics']['precision'],
                'Test_Recall': result['test_metrics']['recall'],
                'Test_F1': result['test_metrics']['f1']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(RESULTS_DIR, f'metrics_summary_report_{timestamp}.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"  ✅ Saved: {summary_file}")
    
    # 3. All Features Report - Detailed list
    print("\n3. Generating All Features Report (Detailed)...")
    
    # Categorize features
    feature_info = []
    for col in feature_cols:
        feature_type = 'Numerical' if pd.api.types.is_numeric_dtype(df[col]) else 'Categorical'
        
        # Determine feature category
        if '_lag' in col:
            category = 'Lag Feature'
            base_feature = col.split('_lag')[0]
            lag_period = col.split('_lag')[1]
            description = f"Value of {base_feature} {lag_period} month(s) ago"
        elif 'rolling_mean' in col:
            category = 'Rolling Mean Feature'
            parts = col.split('_rolling_mean_')
            base_feature = parts[0]
            window = parts[1].replace('m', '')
            description = f"Rolling mean of {base_feature} over {window} month(s)"
        elif 'rolling_std' in col:
            category = 'Rolling Std Feature'
            parts = col.split('_rolling_std_')
            base_feature = parts[0]
            window = parts[1].replace('m', '')
            description = f"Rolling standard deviation of {base_feature} over {window} month(s)"
        elif '_trend' in col:
            category = 'Trend Feature'
            base_feature = col.replace('_trend_3m', '')
            description = f"Trend/slope of {base_feature} over last 3 months"
        elif col in ['obs_year', 'obs_month_num', 'obs_quarter']:
            category = 'Time Feature'
            description = f"Time-based feature: {col}"
        else:
            category = 'Original Feature'
            description = f"Original feature from dataset"
        
        # Get statistics for numerical features
        if feature_type == 'Numerical':
            stats = {
                'Min': df[col].min(),
                'Max': df[col].max(),
                'Mean': df[col].mean(),
                'Std': df[col].std(),
                'Missing_Count': df[col].isnull().sum()
            }
        else:
            stats = {
                'Unique_Values': df[col].nunique(),
                'Missing_Count': df[col].isnull().sum()
            }
        
        feature_info.append({
            'Feature_Name': col,
            'Feature_Type': feature_type,
            'Category': category,
            'Description': description,
            **stats
        })
    
    features_df = pd.DataFrame(feature_info)
    features_file = os.path.join(RESULTS_DIR, f'all_features_report_{timestamp}.csv')
    features_df.to_csv(features_file, index=False)
    print(f"  ✅ Saved: {features_file}")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  - Original features: {len([f for f in feature_cols if '_lag' not in f and 'rolling' not in f and '_trend' not in f and f not in ['obs_year', 'obs_month_num', 'obs_quarter']])}")
    print(f"  - Lag features: {len([f for f in feature_cols if '_lag' in f])}")
    print(f"  - Rolling features: {len([f for f in feature_cols if 'rolling' in f])}")
    print(f"  - Trend features: {len([f for f in feature_cols if '_trend' in f])}")
    print(f"  - Time features: {len([f for f in feature_cols if f in ['obs_year', 'obs_month_num', 'obs_quarter']])}")
    
    return performance_df, summary_df, features_df


def main():
    """Main execution function"""
    # Generate unique timestamp for this run to avoid overwriting files
    RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n{'#'*70}")
    print("Fall Risk Prediction Model - Model 3")
    print(f"{'#'*70}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run timestamp: {RUN_TIMESTAMP}")
    print(f"All output files will be saved with this timestamp to avoid overwriting")
    
    try:
        # Load data
        df = load_data()
        
        # Create targets
        df = create_targets(df)
        
        # Feature engineering
        df = create_features(df)
        
        # Prepare features
        df, feature_cols = prepare_features_for_model(df)
        print(f"\nTotal features for modeling: {len(feature_cols)}")
        
        # Train/test split (70/30)
        train_mask, test_mask = train_test_split(df)
        
        # Train models for each horizon
        all_results = {}
        for horizon in HORIZONS:
            target_col = f'target_{horizon}m'
            
            # Prepare data for this horizon
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            
            # Remove rows with missing target
            valid_mask = y.notna()
            X = X[valid_mask].copy()
            y = y[valid_mask].copy()
            train_mask_valid = train_mask[valid_mask]
            test_mask_valid = test_mask[valid_mask]
            
            # Split data
            X_train = X[train_mask_valid].copy()
            X_test = X[test_mask_valid].copy()
            y_train = y[train_mask_valid].copy()
            y_test = y[test_mask_valid].copy()
            
            print(f"\n{horizon}-month model:")
            print(f"  Training: {len(X_train):,} samples")
            print(f"  Testing: {len(X_test):,} samples")
            print(f"  Target distribution - Train: {y_train.value_counts().to_dict()}")
            print(f"  Target distribution - Test: {y_test.value_counts().to_dict()}")
            
            # Train model
            result = train_model(X_train, y_train, X_test, y_test, horizon, feature_cols)
            all_results[horizon] = result
            
            # Generate ROC-AUC curves
            print(f"\nGenerating ROC-AUC curves for {horizon}-month model...")
            plot_roc_curve(result['y_train'], result['y_train_prob'], horizon, 'Training', RESULTS_DIR, RUN_TIMESTAMP)
            if len(result['y_test']) > 0:
                plot_roc_curve(result['y_test'], result['y_test_prob'], horizon, 'Testing', RESULTS_DIR, RUN_TIMESTAMP)
            else:
                print(f"  ⚠️  Skipping test ROC curve (no test samples)")
            
            # Generate confusion matrices
            print(f"\nGenerating confusion matrices for {horizon}-month model...")
            plot_confusion_matrix(result['y_train'], result['y_train_pred'], horizon, 'Training', RESULTS_DIR, RUN_TIMESTAMP)
            if len(result['y_test']) > 0:
                plot_confusion_matrix(result['y_test'], result['y_test_pred'], horizon, 'Testing', RESULTS_DIR, RUN_TIMESTAMP)
            else:
                print(f"  ⚠️  Skipping test confusion matrix (no test samples)")
        
        # Generate reports
        performance_df, summary_df, features_df = generate_reports(all_results, feature_cols, RUN_TIMESTAMP, df)
        
        print(f"\n{'#'*70}")
        print("✅ All tasks completed successfully!")
        print(f"{'#'*70}")
        print(f"\nSummary:")
        print(f"  - Models trained: {len(all_results)}")
        print(f"  - ROC-AUC curves generated: {len(all_results) * 2} (training + testing)")
        print(f"  - Confusion matrices generated: {len(all_results) * 2} (training + testing)")
        print(f"  - Reports generated: 3")
        print(f"\nResults saved in: {RESULTS_DIR}")
        print(f"Models saved in: {MODELS_DIR}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

