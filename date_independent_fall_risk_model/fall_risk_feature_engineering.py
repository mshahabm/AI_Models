import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


MONTHLY_METRICS = {
    "assist_count",
    "avg_daily_steps",
    "button_press_count",
    "dispatch_cancelled_count",
    "er_dispatch_count",
    "fall_alarm_count",
    "fall_count",
    "help_sent_count",
    "sentiment_negative_count",
    "sentiment_positive_count",
    "sentiment_neutral_count",
    "subscriber_reached_count",
}

COUNT_METRICS = {
    "assist_count",
    "button_press_count",
    "dispatch_cancelled_count",
    "er_dispatch_count",
    "fall_alarm_count",
    "fall_count",
    "help_sent_count",
    "sentiment_negative_count",
    "sentiment_positive_count",
    "sentiment_neutral_count",
    "subscriber_reached_count",
}

MEAN_METRICS = {
    "avg_daily_steps",
}

EVENT_METRICS = [
    "assist_count",
    "fall_alarm_count",
    "fall_count",
    "er_dispatch_count",
    "button_press_count",
    "help_sent_count",
    "subscriber_reached_count",
    "dispatch_cancelled_count",
]

STATIC_OUTPUT_COLUMNS = ["account_number", "age","account_id","health_plan","brand"]

MONTH_COL_PATTERN = re.compile(r"^(?P<metric>.+)_(?P<month>\d{2})_(?P<year>\d{4})$")
LATEST_FILE_PATTERN = re.compile(
    r"^fall_risk_feature_(?P<month>\d{2})_(?P<year>\d{4})\.(parquet|xlsx|xls|csv)$",
    re.IGNORECASE,
)
EPS = 1.0
LOW_STEPS_THRESHOLD = 3000


@dataclass(frozen=True)
class MonthInfo:
    key: str
    timestamp: pd.Timestamp


def _has_wide_monthly_columns(df: pd.DataFrame) -> bool:
    """Check whether dataframe already has <metric>_MM_YYYY style columns."""
    for col in df.columns:
        match = MONTH_COL_PATTERN.match(col)
        if match and match.group("metric") in MONTHLY_METRICS:
            return True
    return False


def _normalize_obs_month_to_key(obs_value: object) -> Optional[str]:
    """
    Normalize long-format month values (e.g., 2025-05, 2025/05, Timestamp)
    into MM_YYYY key.
    """
    if pd.isna(obs_value):
        return None
    ts = pd.to_datetime(str(obs_value), errors="coerce")
    if pd.isna(ts):
        # handle bare YYYY-MM strings robustly
        raw = str(obs_value).strip()
        match = re.match(r"^(?P<year>\d{4})[-_/](?P<month>\d{1,2})$", raw)
        if match:
            mm = int(match.group("month"))
            yyyy = int(match.group("year"))
            return f"{mm:02d}_{yyyy:04d}"
        return None
    return ts.strftime("%m_%Y")


def _convert_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long format:
      account_number, age, obs_month, <metrics...>
    into wide monthly columns:
      <metric>_MM_YYYY
    """
    if "obs_month" not in df.columns:
        raise ValueError(
            "No wide monthly columns detected and 'obs_month' missing; "
            "cannot convert long format."
        )

    work = df.copy()
    work["account_number"] = work["account_number"].astype(str).str.strip()
    work["month_key"] = work["obs_month"].apply(_normalize_obs_month_to_key)
    work = work[work["month_key"].notna()].copy()
    if work.empty:
        raise ValueError("Unable to parse any valid obs_month values from long-format input.")

    # Preserve member ordering from first appearance in input.
    account_order = (
        work["account_number"]
        .drop_duplicates()
        .tolist()
    )

    # Use latest observed age per account.
    age_source = work.copy()
    age_source["obs_ts"] = pd.to_datetime(age_source["month_key"], format="%m_%Y", errors="coerce")
    age_source = age_source.sort_values(["account_number", "obs_ts"])
    latest_age = age_source.groupby("account_number")["age"].last()

    out = pd.DataFrame({"account_number": account_order})
    out["age"] = out["account_number"].map(latest_age).fillna(0)
    out["account_id"] = out["account_number"].map(df.groupby("account_number")["account_id"].first())
    out["health_plan"] = out["account_number"].map(df.groupby("account_number")["health_plan"].first())
    out["brand"] = out["account_number"].map(df.groupby("account_number")["brand"].first())

    for metric in MONTHLY_METRICS:
        if metric not in work.columns:
            continue
        temp = work[["account_number", "month_key", metric]].copy()
        temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
        # Aggregate duplicates per member-month before pivot, preserving NaN semantics:
        # - count-like metrics: sum across duplicate rows
        # - average-like metrics: mean across duplicate rows
        if metric in COUNT_METRICS:
            temp = (
                temp.groupby(["account_number", "month_key"], as_index=False)[metric]
                .sum(min_count=1)
            )
        elif metric in MEAN_METRICS:
            temp = (
                temp.groupby(["account_number", "month_key"], as_index=False)[metric]
                .mean()
            )
        else:
            temp = (
                temp.groupby(["account_number", "month_key"], as_index=False)[metric]
                .first()
            )

        piv = temp.pivot(index="account_number", columns="month_key", values=metric)
        if piv.empty:
            continue
        piv.columns = [f"{metric}_{m}" for m in piv.columns]
        piv = piv.reset_index()
        out = out.merge(piv, on="account_number", how="left")

    return out


def load_input(file_path: str) -> pd.DataFrame:
    """
    Load the source file (parquet/excel/csv), enforce required static columns,
    and preserve row order.
    """
    def _resolve_input_path(raw_path: str) -> str:
        """
        Resolve input path robustly:
        - exact path
        - path with common extensions added
        - same-directory case-insensitive filename match
        """
        if os.path.exists(raw_path):
            return raw_path

        root, ext = os.path.splitext(raw_path)
        candidates: List[str] = []
        if ext:
            candidates.append(raw_path)
        else:
            for candidate_ext in [".parquet", ".xlsx", ".xls", ".csv"]:
                candidates.append(root + candidate_ext)

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        parent_dir = os.path.dirname(raw_path) or "."
        base_name = os.path.basename(raw_path)
        if os.path.isdir(parent_dir):
            entries = os.listdir(parent_dir)
            entry_map = {e.lower(): e for e in entries}
            if base_name.lower() in entry_map:
                return os.path.join(parent_dir, entry_map[base_name.lower()])

            # If caller passed a stem without extension, try stem-only match.
            stem = os.path.splitext(base_name)[0].lower()
            for entry in entries:
                entry_stem, entry_ext = os.path.splitext(entry)
                if entry_stem.lower() == stem and entry_ext.lower() in {".parquet", ".xlsx", ".xls", ".csv"}:
                    return os.path.join(parent_dir, entry)

        raise FileNotFoundError(f"Input file not found: {raw_path}")

    resolved_path = _resolve_input_path(file_path)
    ext = os.path.splitext(resolved_path)[1].lower()

    if ext == ".parquet":
        df = pd.read_parquet(resolved_path)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(resolved_path)
    elif ext == ".csv":
        df = pd.read_csv(resolved_path)
    else:
        raise ValueError(
            f"Unsupported input extension '{ext}'. Supported: .parquet, .xlsx, .xls, .csv"
        )

    print(f"Input file resolved: {resolved_path}")
    print(f"Input format: {ext or 'unknown'}")
    required_static = ["account_number", "age"]
    missing = [c for c in required_static if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required static columns: {missing}")

    # Preserve input ordering and index deterministically.
    df = df.copy().reset_index(drop=True)
    df["account_number"] = df["account_number"].astype(str).str.strip()

    if _has_wide_monthly_columns(df):
        print("Detected wide monthly format (<metric>_MM_YYYY columns).")
        return df

    if "obs_month" in df.columns:
        print("Detected long format (obs_month). Converting to wide monthly columns...")
        converted = _convert_long_to_wide(df)
        print(f"Converted long->wide rows: {len(converted)}, cols: {len(converted.columns)}")
        return converted

    raise ValueError(
        "Input does not contain wide monthly columns or 'obs_month' long-format column."
    )


def detect_latest_monthly_file(input_dir: str) -> str:
    """
    Detect latest monthly file by filename pattern:
      fall_risk_feature_MM_YYYY.<ext>
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    candidates: List[Tuple[int, str]] = []
    for entry in os.listdir(input_dir):
        match = LATEST_FILE_PATTERN.match(entry)
        if not match:
            continue
        mm = int(match.group("month"))
        yyyy = int(match.group("year"))
        sortable = yyyy * 100 + mm
        candidates.append((sortable, os.path.join(input_dir, entry)))

    if not candidates:
        raise FileNotFoundError(
            "No monthly files detected with pattern "
            "'fall_risk_feature_MM_YYYY.<ext>' in: "
            f"{input_dir}"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    latest_file = candidates[0][1]
    print(f"Auto-detected latest monthly file: {latest_file}")
    return latest_file


def extract_month_map(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Parse monthly columns of format <metric>_MM_YYYY and return:
    {
      "MM_YYYY": {
         "<metric>": "<column_name>",
         ...
      },
      ...
    }
    Only configured MONTHLY_METRICS are included.
    """
    month_map: Dict[str, Dict[str, str]] = {}
    for col in df.columns:
        match = MONTH_COL_PATTERN.match(col)
        if not match:
            continue

        metric = match.group("metric")
        if metric not in MONTHLY_METRICS:
            continue

        month_key = f"{match.group('month')}_{match.group('year')}"
        month_map.setdefault(month_key, {})[metric] = col

    if not month_map:
        raise ValueError("No monthly metric columns were detected.")

    return month_map


def _sorted_months(month_map: Dict[str, Dict[str, str]]) -> List[MonthInfo]:
    months: List[MonthInfo] = []
    for key in month_map:
        ts = pd.to_datetime(key, format="%m_%Y")
        months.append(MonthInfo(key=key, timestamp=ts))
    months.sort(key=lambda m: m.timestamp)
    return months


def _safe_series(df: pd.DataFrame, columns: List[str], fill_value: Optional[float] = None) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=df.index)
    temp = df[columns].apply(pd.to_numeric, errors="coerce")
    return temp.fillna(fill_value) if fill_value is not None else temp


def _encode_static(series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    values = series.fillna("UNKNOWN").astype(str)
    categories = sorted(values.unique())
    mapping = {v: i for i, v in enumerate(categories)}
    encoded = values.map(mapping).astype(np.int32)
    return encoded, mapping


def _derive_target_from_label_month(
    df: pd.DataFrame,
    month_map: Dict[str, Dict[str, str]],
    label_month_key: str,
) -> pd.Series:
    """
    Create binary target from explicit label month:
      target_fall_next_month = (fall_count_label_month > 0)
    """
    fall_col = month_map.get(label_month_key, {}).get("fall_count")
    if fall_col is None:
        raise ValueError(
            f"Cannot create target_fall_next_month: fall_count column missing for label month {label_month_key}"
        )
    fall_vals = pd.to_numeric(df[fall_col], errors="coerce")
    # Keep missing outcomes as NaN; do not force missing to 0.
    target = np.where(fall_vals.isna(), np.nan, (fall_vals > 0).astype(float))
    return pd.Series(target, index=df.index, dtype=float)


def build_features(
    df: pd.DataFrame,
    window_months: int,
    include_target: bool = True,
    anchor_month: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build engineered features for the last `window_months` detected months.
    Output contains only static + this window's engineered columns.
    """
    if window_months <= 0:
        raise ValueError("window_months must be > 0")

    month_map = extract_month_map(df)
    sorted_months = _sorted_months(month_map)
    detected_month_keys = [m.key for m in sorted_months]
    if anchor_month:
        anchor_ts = pd.to_datetime(anchor_month.replace("-", "_"), format="%m_%Y", errors="coerce")
        if pd.isna(anchor_ts):
            anchor_ts = pd.to_datetime(anchor_month, format="%Y-%m", errors="coerce")
        if pd.isna(anchor_ts):
            raise ValueError(
                f"Invalid --anchor_month '{anchor_month}'. Use MM_YYYY or YYYY-MM."
            )
        eligible = [m for m in sorted_months if m.timestamp <= anchor_ts]
        if not eligible:
            raise ValueError(
                f"Anchor month {anchor_month} is earlier than all detected months: {detected_month_keys}"
            )
        anchor_info = eligible[-1]
    else:
        anchor_info = sorted_months[-1]

    anchor_idx = next(i for i, m in enumerate(sorted_months) if m.key == anchor_info.key)
    start_idx = max(0, anchor_idx - window_months + 1)
    selected = sorted_months[start_idx:anchor_idx + 1]
    selected_keys = [m.key for m in selected]

    print(f"[window={window_months}] detected months: {detected_month_keys}")
    print(f"[window={window_months}] anchor month: {anchor_info.key}")
    print(f"[window={window_months}] selected months: {selected_keys}")

    out = pd.DataFrame(index=df.index)
    out["account_number"] = df["account_number"]
    out["age"] = pd.to_numeric(df["age"], errors="coerce")
    out["account_id"] = df["account_id"]
    out["health_plan"] = df["health_plan"]
    out["brand"] = df["brand"]

    # ---- Event burden features ----
    for metric in EVENT_METRICS:
        cols = [month_map[m.key].get(metric) for m in selected if metric in month_map[m.key]]
        cols = [c for c in cols if c is not None]
        mdf = _safe_series(df, cols, fill_value=None)

        suffix = f"_last_{window_months}m"
        #out[f"{metric}_sum{suffix}"] = mdf.sum(axis=1, min_count=1)
        #out[f"{metric}_mean{suffix}"] = mdf.mean(axis=1)
        #out[f"{metric}_max{suffix}"] = mdf.max(axis=1)
        #out[f"{metric}_last_value{suffix}"] = mdf.iloc[:, -1] if not mdf.empty else np.nan
        out[f"{metric}_trend{suffix}"] = (
            (mdf.iloc[:, -1] - mdf.iloc[:, 0]) if mdf.shape[1] >= 2 else np.nan
        )
        for i,col in enumerate(mdf.columns):
            j = i + 1
            out[f'{col[:-7]}T-{j}m'] = mdf[col]

    # ---- Mobility features ----
    steps_cols = [month_map[m.key].get("avg_daily_steps") for m in selected if "avg_daily_steps" in month_map[m.key]]
    steps_cols = [c for c in steps_cols if c is not None]
    steps_df = _safe_series(df, steps_cols, fill_value=None)
    suffix = f"_last_{window_months}m"
    #out[f"avg_daily_steps_mean{suffix}"] = steps_df.mean(axis=1)
    #out[f"avg_daily_steps_median{suffix}"] = steps_df.median(axis=1)
    #out[f"avg_daily_steps_std{suffix}"] = steps_df.std(axis=1, ddof=0)
    #out[f"avg_daily_steps_cv{suffix}"] = out[f"avg_daily_steps_std{suffix}"] / (
        #out[f"avg_daily_steps_mean{suffix}"] + EPS
    #)
    out[f"avg_daily_steps_trend{suffix}"] = (
        (steps_df.iloc[:, -1] - steps_df.iloc[:, 0]) if steps_df.shape[1] >= 2 else np.nan
    )
    #out[f"avg_daily_steps_min{suffix}"] = steps_df.min(axis=1)
    #out[f"avg_daily_steps_last_value{suffix}"] = steps_df.iloc[:, -1] if not steps_df.empty else np.nan
    #out[f"avg_daily_steps_max{suffix}"] = steps_df.max(axis=1)
    for i, col in enumerate(steps_df.columns):
        j = i + 1
        out[f'{col[:-7]}T-{j}m'] = steps_df[col]
        
    if not steps_df.empty:
        nonnull_steps = steps_df.notna().sum(axis=1)
        low_steps = ((steps_df < LOW_STEPS_THRESHOLD) & steps_df.notna()).sum(axis=1)
        out[f"low_steps_month_ratio{suffix}"] = np.where(
            nonnull_steps > 0,
            low_steps / nonnull_steps,
            np.nan,
        )
    else:
        out[f"low_steps_month_ratio{suffix}"] = np.nan

    # ---- Sentiment features ----
    neg_cols = [month_map[m.key].get("sentiment_negative_count") for m in selected if "sentiment_negative_count" in month_map[m.key]]
    pos_cols = [month_map[m.key].get("sentiment_positive_count") for m in selected if "sentiment_positive_count" in month_map[m.key]]
    neu_cols = [month_map[m.key].get("sentiment_neutral_count") for m in selected if "sentiment_neutral_count" in month_map[m.key]]
    neg_cols = [c for c in neg_cols if c is not None]
    pos_cols = [c for c in pos_cols if c is not None]
    neu_cols = [c for c in neu_cols if c is not None]

    neg_raw = df[neg_cols].apply(pd.to_numeric, errors="coerce") if neg_cols else pd.DataFrame(index=df.index)
    pos_raw = df[pos_cols].apply(pd.to_numeric, errors="coerce") if pos_cols else pd.DataFrame(index=df.index)
    neu_raw = df[neu_cols].apply(pd.to_numeric, errors="coerce") if neu_cols else pd.DataFrame(index=df.index)

    neg_df = neg_raw
    pos_df = pos_raw
    neu_df = neu_raw

    #neg_sum = neg_df.sum(axis=1, min_count=1)
    #pos_sum = pos_df.sum(axis=1, min_count=1)
    #neu_sum = neu_df.sum(axis=1, min_count=1)
    #sentiment_total = neg_sum + pos_sum + neu_sum

    #out[f"sentiment_total{suffix}"] = sentiment_total
    #out[f"neg_sentiment_share{suffix}"] = neg_sum / (sentiment_total + EPS)
    #out[f"pos_sentiment_share{suffix}"] = pos_sum / (sentiment_total + EPS)
    #out[f"sentiment_polarity{suffix}"] = (pos_sum - neg_sum) / (sentiment_total + EPS)
    #out[f"neg_sentiment_trend{suffix}"] = (
        #(neg_df.iloc[:, -1] - neg_df.iloc[:, 0]) if neg_df.shape[1] >= 2 else 0.0
    #)
    if not neg_df.empty:
        for i, col in enumerate(neg_df.columns):
            j = i + 1
            out[f'{col[:-7]}T-{j}m'] = neg_df[col]
    
    if not pos_df.empty:
        for i, col in enumerate(pos_df.columns):
            j = i + 1
            out[f'{col[:-7]}T-{j}m'] = pos_df[col]

    if not neu_df.empty:
        for i, col in enumerate(neu_df.columns):
            j = i + 1
            out[f'{col[:-7]}T-{j}m'] = neu_df[col]


    if selected:
        availability_count = pd.Series(0, index=df.index, dtype=float)
        for month in selected:
            month_cols = [
                month_map[month.key].get("sentiment_negative_count"),
                month_map[month.key].get("sentiment_positive_count"),
                month_map[month.key].get("sentiment_neutral_count"),
            ]
            month_cols = [c for c in month_cols if c is not None]
            if month_cols:
                month_available = (
                    df[month_cols].apply(pd.to_numeric, errors="coerce").notna().any(axis=1).astype(float)
                )
                availability_count += month_available
        out[f"sentiment_available_month_ratio{suffix}"] = availability_count / len(selected)
    else:
        out[f"sentiment_available_month_ratio{suffix}"] = np.nan

    # ---- Recency-risk features ----
    def months_since_last_nonzero(metric_name: str) -> pd.Series:
        col_list = [month_map[m.key].get(metric_name) for m in selected if metric_name in month_map[m.key]]
        col_list = [c for c in col_list if c is not None]
        if not col_list:
            return pd.Series(np.nan, index=df.index, dtype=float)

        values = _safe_series(df, col_list, fill_value=None).to_numpy(dtype=float)
        recent_index = np.full(shape=(values.shape[0],), fill_value=-1, dtype=int)
        has_any_data = ~np.isnan(values).all(axis=1)
        for j in range(values.shape[1] - 1, -1, -1):
            mask = (recent_index == -1) & (~np.isnan(values[:, j])) & (values[:, j] > 0)
            recent_index[mask] = j
        # 0 means seen in latest month, higher means older, window+1 means never seen.
        result = np.where(recent_index == -1, values.shape[1] + 1, values.shape[1] - 1 - recent_index).astype(float)
        result = np.where(has_any_data, result, np.nan)
        return pd.Series(result, index=df.index, dtype=float)

    #months_since_fall_alarm = months_since_last_nonzero("fall_alarm_count")
    #months_since_fall = months_since_last_nonzero("fall_count")
    #fall_alarm_sum = out[f"fall_alarm_count_sum{suffix}"]
    #er_dispatch_sum = out[f"er_dispatch_count_sum{suffix}"]

    #out[f"months_since_last_fall_alarm{suffix}"] = months_since_fall_alarm
    #out[f"months_since_last_fall{suffix}"] = months_since_fall
    #out[f"had_any_fall_alarm{suffix}"] = (fall_alarm_sum > 0).astype(np.int8)
    #out[f"had_any_er_dispatch{suffix}"] = (er_dispatch_sum > 0).astype(np.int8)

    # ---- Ratios ----
    #assist_sum = out[f"assist_count_sum{suffix}"]
    #dispatch_cancel_sum = out[f"dispatch_cancelled_count_sum{suffix}"]
    #button_sum = out[f"button_press_count_sum{suffix}"]
    #help_sum = out[f"help_sent_count_sum{suffix}"]
    #reached_sum = out[f"subscriber_reached_count_sum{suffix}"]

    #out[f"fall_alarm_to_assist_ratio{suffix}"] = fall_alarm_sum / (assist_sum + EPS)
    #out[f"er_dispatch_to_fall_alarm_ratio{suffix}"] = er_dispatch_sum / (fall_alarm_sum + EPS)
    #out[f"dispatch_cancelled_rate{suffix}"] = dispatch_cancel_sum / (er_dispatch_sum + dispatch_cancel_sum + EPS)
    #out[f"button_press_to_fall_alarm_ratio{suffix}"] = button_sum / (fall_alarm_sum + EPS)
    #out[f"help_to_reached_ratio{suffix}"] = help_sum / (reached_sum + EPS)

    # ---- Data quality / sparsity ----
    metric_month_columns: List[str] = []
    for month in selected:
        for metric in MONTHLY_METRICS:
            col = month_map[month.key].get(metric)
            if col is not None:
                metric_month_columns.append(col)

    if metric_month_columns:
        metric_raw = df[metric_month_columns].apply(pd.to_numeric, errors="coerce")
        out[f"data_completeness{suffix}"] = metric_raw.notna().mean(axis=1)
    else:
        out[f"data_completeness{suffix}"] = np.nan

    months_present = pd.Series(0, index=df.index, dtype=float)
    active_months = pd.Series(0, index=df.index, dtype=float)
    activity_metrics = [
        "assist_count",
        "fall_alarm_count",
        "fall_count",
        "er_dispatch_count",
        "button_press_count",
        "help_sent_count",
        "subscriber_reached_count",
    ]

    for month in selected:
        month_cols = [month_map[month.key].get(m) for m in MONTHLY_METRICS if month_map[month.key].get(m) is not None]
        if month_cols:
            month_data = df[month_cols].apply(pd.to_numeric, errors="coerce")
            month_present = month_data.notna().any(axis=1).astype(float)
            months_present += month_present

        activity_cols = [month_map[month.key].get(m) for m in activity_metrics if month_map[month.key].get(m) is not None]
        if activity_cols:
            activity_data = df[activity_cols].apply(pd.to_numeric, errors="coerce")
            month_active = ((activity_data > 0) & activity_data.notna()).any(axis=1).astype(float)
            active_months += month_active

    out[f"months_present{suffix}"] = months_present
    out[f"active_months{suffix}"] = active_months
    out[f"all_zero_activity_flag{suffix}"] = (active_months == 0).astype(np.int8)

    # Fill non-numeric edge cases only; keep numeric NaN as missing values.
    for col in out.columns:
        if col == "account_number":
            continue
        if out[col].dtype.kind not in "iufb":
            out[col] = out[col].fillna("UNKNOWN")

    # Preserve the original account_number row order and ensure uniqueness in output.
    duplicate_count = int(out["account_number"].duplicated(keep=False).sum())
    if duplicate_count > 0:
        print(f"[window={window_months}] duplicate key check: found {duplicate_count} duplicate rows")
        out = out.drop_duplicates(subset=["account_number"], keep="first")
    else:
        print(f"[window={window_months}] duplicate key check: no duplicates")

    if include_target:
        next_idx = anchor_idx + 1
        if next_idx >= len(sorted_months):
            raise ValueError(
                f"Cannot derive target_fall_next_month: no label month available after anchor {anchor_info.key}."
            )
        label_month_key = sorted_months[next_idx].key
        print(f"[window={window_months}] label month for target: {label_month_key}")
        out["target_fall_next_month"] = _derive_target_from_label_month(
            df=df,
            month_map=month_map,
            label_month_key=label_month_key,
        )
        tgt = out["target_fall_next_month"]
        print(
            f"[window={window_months}] target stats: "
            f"non_null={int(tgt.notna().sum())}, "
            f"positive={int((tgt == 1).sum())}, "
            f"negative={int((tgt == 0).sum())}"
        )

    # Keep static columns first; everything else belongs to this window only.
    feature_cols = [c for c in out.columns if c not in STATIC_OUTPUT_COLUMNS]
    out = out[STATIC_OUTPUT_COLUMNS + feature_cols]

    print(f"[window={window_months}] row count: {len(out)}")
    print(f"[window={window_months}] total feature count: {len(out.columns)}")
    return out


def build_and_save_window(
    df: pd.DataFrame,
    window_months: int,
    output_dir: str,
    include_target: bool = True,
    anchor_month: Optional[str] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    features = build_features(
        df,
        window_months=window_months,
        include_target=include_target,
        anchor_month=anchor_month,
    )
    file_name = f"fall_risk_feature_last_{window_months}m.parquet"
    output_path = os.path.join(output_dir, file_name)
    features.to_parquet(output_path, index=False)
    print(f"[window={window_months}] output file path: {output_path}")
    return output_path


def build_and_save_window_variant(
    df: pd.DataFrame,
    window_months: int,
    output_dir: str,
    include_target: bool,
    file_suffix: str = "",
    anchor_month: Optional[str] = None,
) -> str:
    """
    Save a specific variant of the same window:
      - scoring variant (features only)
      - training variant (features + target)
    """
    os.makedirs(output_dir, exist_ok=True)
    features = build_features(
        df,
        window_months=window_months,
        include_target=include_target,
        anchor_month=anchor_month,
    )
    file_name = f"fall_risk_feature_last_{window_months}m{file_suffix}.parquet"
    output_path = os.path.join(output_dir, file_name)
    features.to_parquet(output_path, index=False)
    variant = "training" if include_target else "scoring"
    print(f"[window={window_months}] {variant} output file path: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature engineering for fall-risk rollover windows.")
    parser.add_argument(
        "--input_file",
        default="202501_to_202603_health.parquet",
        help=(
            "Path to wide monthly input file (.parquet/.xlsx/.xls/.csv). "
            "Use 'auto' to detect latest fall_risk_feature_MM_YYYY file."
        ),
    )
    parser.add_argument(
        "--input_dir",
        default=".",
        help="Directory used for --input_file auto detection.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory to write window-specific parquet feature files.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[6, 4, 3],
        help="Rollover windows (months), e.g. --windows 6 4 3",
    )
    parser.add_argument(
        "--anchor_month",
        default=None,
        help=(
            "Optional anchor month for feature window end. "
            "Accepted formats: MM_YYYY or YYYY-MM. "
            "If omitted, uses latest detected month."
        ),
    )
    parser.add_argument(
        "--include_target",
        action="store_true",
        help="Include target_fall_next_month derived from latest month's fall_count.",
    )
    parser.add_argument(
        "--save_both_versions",
        action="store_true",
        help=(
            "Generate both versions per window: "
            "features-only (scoring) and with-target (training)."
        ),
    )
    parser.add_argument(
        "--train_output_dir",
        default=None,
        help=(
            "Optional training output directory used with --save_both_versions. "
            "Defaults to <output_dir>/train."
        ),
    )
    parser.add_argument(
        "--score_output_dir",
        default=None,
        help=(
            "Optional scoring output directory used with --save_both_versions. "
            "Defaults to <output_dir>/score."
        ),
    )

    args = parser.parse_args()
    resolved_input: Optional[str] = args.input_file
    if args.input_file.strip().lower() == "auto":
        resolved_input = detect_latest_monthly_file(args.input_dir)

    df = load_input(resolved_input)
    unique_windows = sorted(set(args.windows), reverse=True)

    print(f"Input rows: {len(df)}")
    print(f"Requested windows: {unique_windows}")
    print(f"Include target_fall_next_month: {args.include_target}")
    print(f"Save both versions: {args.save_both_versions}")
    print(f"Anchor month: {args.anchor_month if args.anchor_month else 'latest'}")

    for w in unique_windows:
        if args.save_both_versions:
            score_dir = args.score_output_dir or os.path.join(args.output_dir, "score")
            train_dir = args.train_output_dir or os.path.join(args.output_dir, "train")
            build_and_save_window_variant(
                df=df,
                window_months=w,
                output_dir=score_dir,
                include_target=False,
                file_suffix="",
                anchor_month=args.anchor_month,
            )
            build_and_save_window_variant(
                df=df,
                window_months=w,
                output_dir=train_dir,
                include_target=True,
                file_suffix="_with_target",
                anchor_month=args.anchor_month,
            )
        else:
            build_and_save_window(
                df=df,
                window_months=w,
                output_dir=args.output_dir,
                include_target=args.include_target,
                anchor_month=args.anchor_month,
            )


if __name__ == "__main__":
    main()
