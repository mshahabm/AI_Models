"""
Fall Risk Scoring - 6 Month Rollover (Inference Only)

This script scores member-level fall risk using a pre-trained model and an
engineered 6-month feature file. It reuses QA-approved model logic from
FallRisk_Healthplans_Incremental.py and outputs only:
  - account_number
  - Risk_Score
  - Risk_Category
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

from FallRisk_Healthplans_Incremental import FallRiskForecastingModel, get_risk_category


def detect_file_format(file_path_base: str) -> str:
    """Detect CSV or Parquet file when extension is omitted."""
    base = file_path_base.rsplit(".", 1)[0] if "." in file_path_base else file_path_base
    for ext in [".parquet", ".csv", ".CSV"]:
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate
    return file_path_base


def read_dataframe(file_path: str) -> pd.DataFrame:
    """Read feature dataframe from Parquet or CSV."""
    actual_path = detect_file_format(file_path)
    if not os.path.exists(actual_path):
        raise FileNotFoundError(f"Feature file not found: {file_path}")
    if actual_path.lower().endswith(".parquet"):
        print(f"Reading Parquet: {actual_path}")
        return pd.read_parquet(actual_path)
    print(f"Reading CSV: {actual_path}")
    return pd.read_csv(actual_path)


def save_dataframe(df: pd.DataFrame, output_path: str) -> str:
    """Save output dataframe to parquet/csv based on extension."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if output_path.lower().endswith(".parquet"):
        df.to_parquet(output_path, index=False)
    else:
        if not output_path.lower().endswith(".csv"):
            output_path = output_path + ".csv"
        df.to_csv(output_path, index=False)
    return output_path


def _score_from_rank(values: pd.Series, min_score: int, max_score: int) -> pd.Series:
    """
    Map values to integer score band [min_score, max_score] using percentile rank.
    This keeps risk-score buckets meaningful even when threshold shifts month-to-month.
    """
    if values.empty:
        return pd.Series(dtype=int)
    if min_score == max_score:
        return pd.Series(min_score, index=values.index, dtype=int)

    ranks = values.rank(method="average", pct=True).clip(lower=0.0, upper=1.0)
    span = max_score - min_score + 1
    scores = (np.ceil(ranks * span) + (min_score - 1)).clip(lower=min_score, upper=max_score)
    return scores.astype(int)


def assign_risk_scores_threshold_aware(probabilities: pd.Series, threshold: float) -> pd.Series:
    """
    Threshold-aware dynamic mapping:
      - probs < 0.5*threshold               -> scores 1..2  (Low)
      - 0.5*threshold <= probs < 1.75*threshold -> scores 3..6  (Moderate)
      - probs >= 1.75*threshold            -> scores 7..10 (High)

    Category mapping remains unchanged:
      Low: 1..2, Medium: 3..6, High: 7..10
    """
    probs = pd.to_numeric(probabilities, errors="coerce")
    low_cut = 0.8 * threshold
    high_cut = 2.75 * threshold
    low_mask = probs < low_cut
    medium_mask = (probs >= low_cut) & (probs < high_cut)
    high_mask = probs >= high_cut

    scores = pd.Series(index=probs.index, dtype=int)
    low_scores = _score_from_rank(probs[low_mask], min_score=1, max_score=2)
    medium_scores = _score_from_rank(probs[medium_mask], min_score=3, max_score=6)
    high_scores = _score_from_rank(probs[high_mask], min_score=7, max_score=10)

    scores.loc[low_scores.index] = low_scores
    scores.loc[medium_scores.index] = medium_scores
    scores.loc[high_scores.index] = high_scores
    return scores.fillna(1).astype(int)


def build_report(
    feature_file: str,
    model_file: str,
    predict_month: str,
    output_file: str,
) -> str:
    """Load features + model, score records, and write compact report."""
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    df = read_dataframe(feature_file)
    if "account_number" not in df.columns:
        raise ValueError("Input feature file must contain 'account_number'.")
    if "target_fall_next_month" in df.columns:
        print("WARNING: Dropping target_fall_next_month from scoring input.")
        df = df.drop(columns=["target_fall_next_month"])

    model = FallRiskForecastingModel.load(model_file)
    print(f"Loaded model: {model_file}")
    print(f"Scoring rows: {len(df):,}")

    missing_features = [f for f in model.feature_names if f not in df.columns]
    if missing_features:
        preview = ", ".join(missing_features[:10])
        raise ValueError(
            f"Input features missing {len(missing_features)} required model columns. "
            f"First missing: {preview}"
        )

    probabilities = model.predict_proba(df)
    threshold = model.get_adaptive_threshold(predict_month)
    print(f"Using adaptive threshold for {predict_month}: {threshold:.4f}")
    risk_scores = assign_risk_scores_threshold_aware(
        probabilities=pd.Series(probabilities),
        threshold=threshold,
    )

    report_df = pd.DataFrame(
        {
            "account_number": df["account_number"].astype(str).str.strip(),
            "account_id": df["account_id"],
            "age": df["age"],
            "brand": df["brand"],
            "health_plan": df["health_plan"],
            "member name": df.get("member name",''),
            "care manager": df.get("care_manager",''),
            "Risk_Score": risk_scores.values
        }
    )
    report_df["Risk_Category"] = report_df["Risk_Score"].apply(get_risk_category)

    output_path = save_dataframe(report_df, output_file)
    print(f"Saved report: {output_path}")
    print(f"Members scored: {len(report_df):,}")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference-only scoring for 6-month rollover fall risk model."
    )
    parser.add_argument(
        "--feature_file",
        type=str,
        default="fall_risk_features_last_6m.parquet",
        help="Path to 6m engineered feature file (parquet/csv).",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        required=True,
        help="Path to saved model pickle generated by training pipeline.",
    )
    parser.add_argument(
        "--predict_month",
        type=str,
        default="05_2025",
        help="Prediction month in MM_YYYY format for adaptive threshold lookup.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="Fall_Risk_score_052025_6m.parquet",
        help="Output report path (.parquet or .csv).",
    )
    parser.add_argument(
        "--etl_file",
        type=str,
        default="202501_to_202603_health.parquet",
        help="(Optional) Path to raw health data for on-the-fly feature engineering."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        build_report(
            feature_file=args.feature_file,
            model_file=args.model_file,
            predict_month=args.predict_month,
            output_file=args.output_file,
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
