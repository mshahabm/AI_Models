"""Extract monthly feature data from parquet/CSV by obs_month. Output: Fallrisk_Test_MM_YYYY_Healthplans.parquet"""
import argparse
import os
import sys
import pandas as pd

try:
    import pyarrow
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

ID_COLS = ['account_number', 'account_id', 'age', 'health_plan', 'brand']
INPUT_FILE = "healthplan_base_features.parquet"
MONTH = "2026-01"
OUTPUT_BASE = "Fallrisk_Test_{suffix}_Healthplans"


def _normalize_month(s):
    s = str(s).strip().replace('_', '-').split('-')
    if len(s) != 2:
        raise ValueError("Month must be YYYY-MM or MM-YYYY")
    a, b = s[0], s[1]
    return f"{a}-{b}" if len(a) == 4 else f"{b}-{a}"


def _month_suffix(yyyy_mm):
    p = yyyy_mm.split('-')
    return f"{p[1]}_{p[0]}" if len(p) == 2 else yyyy_mm.replace('-', '_')


def _obs_to_yyyy_mm(ser):
    if pd.api.types.is_integer_dtype(ser) or pd.api.types.is_float_dtype(ser):
        s = ser.dropna().astype(int)
        r = (s // 100).astype(str) + '-' + (s % 100).astype(str).str.zfill(2)
        return r.reindex(ser.index).where(ser.notna())
    if isinstance(ser.dtype, pd.PeriodDtype):
        return ser.astype(str)
    return pd.to_datetime(ser, errors='coerce').dt.strftime('%Y-%m')


def _path(file_path):
    base = file_path.rsplit('.', 1)[0] if '.' in file_path else file_path
    for ext in ['.CSV', '.csv', '.parquet']:
        if os.path.exists(base + ext):
            return base + ext
    return file_path


def _read(path, filters=None):
    p = _path(path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"File not found: {path}")
    if p.lower().endswith('.parquet'):
        if not HAS_PARQUET:
            raise ImportError("pyarrow required for parquet")
        return pd.read_parquet(p, filters=filters)
    return pd.read_csv(p)


def extract_monthly_features(input_path, month_arg, output_suffix=None, output_dir=None):
    yyyy_mm = _normalize_month(month_arg)
    suffix = _month_suffix(yyyy_mm)
    filt_int = int(yyyy_mm[:4]) * 100 + int(yyyy_mm[5:7])

    if _path(input_path).lower().endswith('.parquet') and HAS_PARQUET:
        try:
            df = _read(input_path, filters=[('obs_month', '==', filt_int)])
        except Exception:
            df = _read(input_path)
        if df.empty:
            df = _read(input_path)
    else:
        df = _read(input_path)

    if 'obs_month' not in df.columns:
        raise ValueError("Input must have 'obs_month'")
    if 'account_number' not in df.columns and 'account_number_id' in df.columns:
        df['account_number'] = df['account_number_id'].astype(str)
    if 'account_number' not in df.columns:
        raise ValueError("Input must have 'account_number' or 'account_number_id'")

    df['_m'] = _obs_to_yyyy_mm(df['obs_month'])
    sub = df.loc[df['_m'] == yyyy_mm].drop(columns=['_m'])
    if sub.empty:
        raise ValueError(f"No rows for obs_month = {yyyy_mm}")

    id_cols = [c for c in ID_COLS if c in sub.columns]
    if 'account_number' not in id_cols:
        id_cols = ['account_number'] + [c for c in id_cols if c != 'account_number']
    feature_cols = [c for c in sub.columns if c not in id_cols and c != 'obs_month']
    ren = {c: f"{c}_{suffix}" for c in feature_cols}
    result = sub[id_cols + feature_cols].rename(columns=ren).drop_duplicates(subset=['account_number'], keep='first').reset_index(drop=True)

    out_dir = output_dir or os.path.dirname(os.path.abspath(_path(input_path)))
    os.makedirs(out_dir, exist_ok=True)
    base = OUTPUT_BASE.format(suffix=suffix) + (f"_{output_suffix}" if output_suffix else "")

    if not HAS_PARQUET:
        raise ImportError("pyarrow required for parquet output. Install: pip install pyarrow")
    pq_p = os.path.join(out_dir, f"{base}.parquet")
    result.to_parquet(pq_p, index=False, engine='pyarrow')
    print(f"  Wrote {pq_p}")
    print(f"  Rows: {len(result):,} | Cols: {len(result.columns)}")
    return pq_p


def main():
    p = argparse.ArgumentParser(description="Extract monthly features -> Fallrisk_Test_MM_YYYY_Healthplans.parquet")
    p.add_argument("input_file", nargs="?", default=INPUT_FILE, help="Input parquet/CSV")
    p.add_argument("month", nargs="?", default=MONTH, help="Month: YYYY-MM or MM-YYYY")
    p.add_argument("--suffix", default=None)
    p.add_argument("-o", "--output-dir", default=None)
    args = p.parse_args()
    if not args.input_file or not args.month:
        p.error("Provide input_file and month (in script or CLI)")
    try:
        out = extract_monthly_features(args.input_file, args.month, args.suffix, args.output_dir)
        print(f"Done: {out}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

