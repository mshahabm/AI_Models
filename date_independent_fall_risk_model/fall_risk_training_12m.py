"""
Fall Risk Training - 6 Month Rollover (Training/Validation Only)

This script trains a model from an engineered 6-month feature dataset that
already includes a binary target column for next-month fall outcome.

Outputs:
  - Trained model artifact (.pkl) compatible with fall_Risk_score_6m.py
  - Validation metrics and plots in output directory
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score

from FallRisk_Healthplans_Incremental import (
    FallRiskForecastingModel,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_auc,
    save_confusion_matrix_txt,
)


def detect_file_format(file_path_base: str) -> str:
    """Detect CSV or Parquet file when extension is omitted."""
    base = file_path_base.rsplit(".", 1)[0] if "." in file_path_base else file_path_base
    for ext in [".parquet", ".csv", ".CSV"]:
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate
    return file_path_base


def read_dataframe(file_path: str) -> pd.DataFrame:
    """Read training dataframe from Parquet or CSV."""
    actual_path = detect_file_format(file_path)
    if not os.path.exists(actual_path):
        raise FileNotFoundError(f"Training file not found: {file_path}")
    if actual_path.lower().endswith(".parquet"):
        print(f"Reading Parquet: {actual_path}")
        return pd.read_parquet(actual_path)
    print(f"Reading CSV: {actual_path}")
    return pd.read_csv(actual_path)


def _infer_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """Infer feature columns by excluding identifiers and target."""
    exclude_cols = {
        target_col,
        "account_number",
        "account_id",
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding ID/target columns.")
    return feature_cols


def _sanitize_target(y: pd.Series, target_col: str) -> pd.Series:
    """Ensure binary numeric target values."""
    y_numeric = pd.to_numeric(y, errors="coerce")
    if y_numeric.isna().any():
        raise ValueError(
            f"Target column '{target_col}' contains null or non-numeric values "
            "after labeled-row filtering. Expected binary 0/1 values."
        )
    unique_vals = sorted(y_numeric.unique().tolist())
    if not set(unique_vals).issubset({0, 1}):
        raise ValueError(
            f"Target column '{target_col}' must be binary 0/1. Found values: {unique_vals}"
        )
    return y_numeric.astype(int)


def _select_threshold_with_error_caps(
    y_true: pd.Series,
    y_proba: np.ndarray,
    max_fpr: float = 0.30,
    max_fnr: float = 0.25,
    min_threshold: float = 0.03,
    fpr_weight: float = 3.0,
    fnr_weight: float = 1.0,
) -> tuple[float, dict]:
    """
    Select threshold closest to desired operating point:
      - target FPR ~= max_fpr
      - target FNR ~= max_fnr

    This avoids unstable behavior where optimizing one error can explode the other.
    """
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(y_proba).astype(float)

    # Candidate thresholds from score distribution (+edges).
    unique_scores = np.unique(np.round(p_arr, 6))
    thresholds = np.concatenate(([0.0], unique_scores, [1.0]))
    thresholds = np.unique(thresholds)
    thresholds = thresholds[thresholds >= float(min_threshold)]
    if thresholds.size == 0:
        thresholds = np.array([float(min_threshold)])

    best = None  # closest to target operating point
    for thr in thresholds:
        y_pred = (p_arr >= thr).astype(int)
        cm = confusion_matrix(y_arr, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        rec = recall_score(y_arr, y_pred, zero_division=0)
        prec = precision_score(y_arr, y_pred, zero_division=0)
        entry = {
            "threshold": float(thr),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "recall": float(rec),
            "precision": float(prec),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

        # Distance to target operating point (around max_fpr / max_fnr).
        entry["target_distance"] = (
            fpr_weight * abs(fpr - max_fpr) +
            fnr_weight * abs(fnr - max_fnr)
        )

        if (
            best is None
            or entry["target_distance"] < best["target_distance"]
            or (
                entry["target_distance"] == best["target_distance"]
                and abs(entry["fpr"] - max_fpr) < abs(best["fpr"] - max_fpr)
            )
            or (
                entry["target_distance"] == best["target_distance"]
                and abs(entry["fpr"] - max_fpr) == abs(best["fpr"] - max_fpr)
                and entry["precision"] > best["precision"]
            )
            or (
                entry["target_distance"] == best["target_distance"]
                and abs(entry["fpr"] - max_fpr) == abs(best["fpr"] - max_fpr)
                and entry["precision"] == best["precision"]
                and entry["threshold"] > best["threshold"]
            )
        ):
            best = entry

    chosen = best
    chosen["selection_mode"] = "target_distance"
    return float(chosen["threshold"]), chosen


def train_model(
    train_file: str,
    target_col: str,
    output_dir: str,
    model_file: str,
    predict_month: str,
    algorithm: str,
    val_size: float,
    random_state: int,
    max_validation_fpr: float,
    max_validation_fnr: float,
    min_threshold: float,
    fpr_weight: float,
    fnr_weight: float,
    retrain_model: str | None,
) -> str:
    """Train model, validate, save artifacts, and return model path."""
    os.makedirs(output_dir, exist_ok=True)

    df = read_dataframe(train_file)
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in input file. "
            "Training requires features + binary target."
        )

    # Keep only labeled outcomes. Feature engineering now preserves missing targets
    # for members without next-month outcomes, which must be excluded from training.
    target_numeric = pd.to_numeric(df[target_col], errors="coerce")
    labeled_mask = target_numeric.notna()
    unlabeled_count = int((~labeled_mask).sum())
    if unlabeled_count > 0:
        print(f"Dropping unlabeled rows (missing {target_col}): {unlabeled_count:,}")
    df_labeled = df.loc[labeled_mask].reset_index(drop=True)
    if df_labeled.empty:
        raise ValueError(
            f"No labeled rows found in '{target_col}'. "
            "Cannot train without at least some non-null labels."
        )

    feature_cols = _infer_feature_columns(df_labeled, target_col)
    X = df_labeled[feature_cols].copy()
    y = _sanitize_target(df_labeled[target_col], target_col)

    print(f"Rows (input): {len(df):,}")
    print(f"Rows (labeled): {len(df_labeled):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target positives: {int(y.sum()):,} ({(y.mean() * 100):.2f}%)")
    if len(np.unique(y)) < 2:
        raise ValueError(
            f"Target column '{target_col}' has only one class after filtering. "
            "Need both 0 and 1 to train a classifier."
        )

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size, random_state=random_state
    )
    idx = np.arange(len(df_labeled))
    train_idx, val_idx = next(splitter.split(idx, y))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)

    print(f"Train rows: {len(X_train):,}, positives: {int(y_train.sum()):,} ({(y_train.mean() * 100):.2f}%)")
    print(f"Val rows: {len(X_val):,}, positives: {int(y_val.sum()):,} ({(y_val.mean() * 100):.2f}%)")

    if retrain_model:
        print(f"Loading existing model for retraining: {retrain_model}")
        model = FallRiskForecastingModel.load(retrain_model)
        # Update algorithm if changed (though typically should match)
        model.algorithm = algorithm
    else:
        model = FallRiskForecastingModel(
            algorithm=algorithm,
            use_temporal_features=True,
            use_smote=False,
        )
    optimal_threshold = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        predict_month=predict_month,
    )

    y_val_proba = model.predict_proba(X_val)
    # QA-safe threshold update using validation only (no test leakage).
    capped_threshold, threshold_stats = _select_threshold_with_error_caps(
        y_true=y_val,
        y_proba=y_val_proba,
        max_fpr=max_validation_fpr,
        max_fnr=max_validation_fnr,
        min_threshold=min_threshold,
        fpr_weight=fpr_weight,
        fnr_weight=fnr_weight,
    )
    model.adaptive_thresholds[predict_month] = capped_threshold
    print(
        "Validation threshold update "
        f"(max FPR={max_validation_fpr:.2f}, max FNR={max_validation_fnr:.2f}, min_thr={min_threshold:.3f}, "
        f"fpr_w={fpr_weight:.2f}, fnr_w={fnr_weight:.2f}): "
        f"mode={threshold_stats.get('selection_mode', 'unknown')}, "
        f"threshold={capped_threshold:.4f}, "
        f"FPR={threshold_stats['fpr']:.4f}, "
        f"FNR={threshold_stats['fnr']:.4f}, "
        f"dist={threshold_stats.get('target_distance', np.nan):.4f}, "
        f"Recall={threshold_stats['recall']:.4f}, "
        f"Precision={threshold_stats['precision']:.4f}"
    )

    y_val_pred = model.predict(X_val, predict_month=predict_month)
    val_auc = roc_auc_score(y_val, y_val_proba) if len(np.unique(y_val)) > 1 else 0.0
    print(f"Validation ROC-AUC: {val_auc:.4f}")

    # Validation reports
    plot_roc_auc(
        y_val,
        y_val_proba,
        f"ROC-AUC Validation - 6m ({predict_month})",
        os.path.join(output_dir, "ROC_AUC_Val.png"),
    )
    plot_confusion_matrix(
        y_val,
        y_val_pred,
        f"CM Validation - 6m ({predict_month})",
        os.path.join(output_dir, "CM_Val.png"),
    )
    save_confusion_matrix_txt(
        y_val,
        y_val_pred,
        y_val_proba,
        os.path.join(output_dir, "CM_Val.txt"),
        f"Validation 6m {predict_month}",
    )
    evaluate_model(
        y_val,
        y_val_pred,
        y_val_proba,
        os.path.join(output_dir, "Performance_Val.txt"),
    )

    model.save(model_file)
    print(f"Saved model: {model_file}")

    metadata = pd.DataFrame(
        [
            {"key": "train_file", "value": train_file},
            {"key": "target_col", "value": target_col},
            {"key": "predict_month", "value": predict_month},
            {"key": "algorithm", "value": algorithm},
            {"key": "val_size", "value": str(val_size)},
            {"key": "random_state", "value": str(random_state)},
            {"key": "n_rows_input", "value": str(len(df))},
            {"key": "n_rows_labeled", "value": str(len(df_labeled))},
            {"key": "n_rows_unlabeled", "value": str(unlabeled_count)},
            {"key": "n_features", "value": str(len(feature_cols))},
            {"key": "train_positive_rate", "value": f"{y_train.mean():.6f}"},
            {"key": "val_positive_rate", "value": f"{y_val.mean():.6f}"},
            {"key": "val_roc_auc", "value": f"{val_auc:.6f}"},
            {"key": "threshold_predict_month", "value": f"{capped_threshold:.6f}"},
            {"key": "threshold_fpr", "value": f"{threshold_stats['fpr']:.6f}"},
            {"key": "threshold_fnr", "value": f"{threshold_stats['fnr']:.6f}"},
            {"key": "threshold_recall", "value": f"{threshold_stats['recall']:.6f}"},
            {"key": "threshold_precision", "value": f"{threshold_stats['precision']:.6f}"},
            {"key": "max_validation_fpr", "value": f"{max_validation_fpr:.6f}"},
            {"key": "max_validation_fnr", "value": f"{max_validation_fnr:.6f}"},
            {"key": "min_threshold", "value": f"{min_threshold:.6f}"},
            {"key": "fpr_weight", "value": f"{fpr_weight:.6f}"},
            {"key": "fnr_weight", "value": f"{fnr_weight:.6f}"},
            {"key": "threshold_target_distance", "value": f"{threshold_stats.get('target_distance', np.nan):.6f}"},
            {"key": "threshold_selection_mode", "value": str(threshold_stats.get("selection_mode", "unknown"))},
        ]
    )
    metadata.to_csv(os.path.join(output_dir, "Training_Metadata.csv"), index=False)

    return model_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training/validation pipeline for 6-month rollover fall risk model."
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="fall_risk_features_last_6m_with_target.parquet",
        help="Path to 6m engineered training file containing target column.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="target_fall_next_month",
        help="Binary target column name (0/1).",
    )
    parser.add_argument(
        "--predict_month",
        type=str,
        default="05_2025",
        help="Prediction month in MM_YYYY format for adaptive threshold storage.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["XGBoost", "RandomForest", "GradientBoosting", "LogisticRegression"],
        default="XGBoost",
        help="Model algorithm.",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.3,
        help="Validation split fraction.",
    )
    parser.add_argument(
        "--max_validation_fpr",
        type=float,
        default=0.30,
        help="Maximum allowed validation FPR when selecting decision threshold.",
    )
    parser.add_argument(
        "--max_validation_fnr",
        type=float,
        default=0.25,
        help="Maximum allowed validation FNR when selecting decision threshold.",
    )
    parser.add_argument(
        "--min_threshold",
        type=float,
        default=0.03,
        help="Minimum decision threshold floor to avoid ultra-low cutoffs.",
    )
    parser.add_argument(
        "--fpr_weight",
        type=float,
        default=3.0,
        help="Penalty weight for FPR-cap violation in weighted tradeoff fallback.",
    )
    parser.add_argument(
        "--fnr_weight",
        type=float,
        default=1.0,
        help="Penalty weight for FNR-cap violation in weighted tradeoff fallback.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for stratified split.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="FallRisk_6m_Training_Output",
        help="Directory to save validation outputs and metadata.",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="fall_risk_model_6m.pkl",
        help="Output path for saved model artifact (.pkl).",
    )
    parser.add_argument(
        "--retrain_model",
        type=str,
        default=None,
        help="Path to existing model file (.pkl) to load and retrain. If not provided, trains a new model from scratch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        train_model(
            train_file=args.train_file,
            target_col=args.target_col,
            output_dir=args.output_dir,
            model_file=args.model_file,
            predict_month=args.predict_month,
            algorithm=args.algorithm,
            val_size=args.val_size,
            random_state=args.random_state,
            max_validation_fpr=args.max_validation_fpr,
            max_validation_fnr=args.max_validation_fnr,
            min_threshold=args.min_threshold,
            fpr_weight=args.fpr_weight,
            fnr_weight=args.fnr_weight,
            retrain_model=args.retrain_model,
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
