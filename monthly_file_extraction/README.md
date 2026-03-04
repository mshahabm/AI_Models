# Monthly File Extraction

Use `test_file_automation.py` to extract records for a single month from a base
CSV/parquet file and write a month-suffixed parquet output.

## Requirements

- Python 3.9+
- `pandas`
- `pyarrow` (required for parquet input/output)

Install:

```bash
pip install pandas pyarrow
```

## Input Expectations

Input file can be `.parquet` or `.csv`. The script accepts:

- `obs_month` (required)
- `account_number` (required), or `account_number_id` (auto-mapped to
  `account_number`)

Optional ID columns preserved when available:

- `account_id`
- `age`
- `health_plan`
- `brand`

`obs_month` can be represented as:

- Integer like `202601`
- Date/time-like values parseable by pandas
- Period-like values

## Usage

From this folder:

```bash
python test_file_automation.py [input_file] [month] [--suffix TEXT] [-o OUTPUT_DIR]
```

Defaults if omitted:

- `input_file`: `healthplan_base_features.parquet`
- `month`: `2026-01`

Accepted month formats:

- `YYYY-MM` (example: `2026-01`)
- `MM-YYYY` (example: `01-2026`)

## Example Commands

Use defaults:

```bash
python test_file_automation.py
```

Specify file and month:

```bash
python test_file_automation.py healthplan_base_features.parquet 2026-01
```

Add a custom suffix and output directory:

```bash
python test_file_automation.py healthplan_base_features.parquet 2026-01 --suffix v2 -o ./out
```

## Output

Output file name pattern:

- `Fallrisk_Test_MM_YYYY_Healthplans.parquet`
- If `--suffix` is provided:
  `Fallrisk_Test_MM_YYYY_Healthplans_<suffix>.parquet`

Behavior details:

- Keeps one row per `account_number` (`drop_duplicates`, first row kept)
- Leaves `obs_month` unchanged
- Renames non-ID feature columns with `_<MM_YYYY>` suffix
- Writes parquet using `pyarrow`

## Common Errors

- `File not found: ...`: input path or extension does not exist
- `Input must have 'obs_month'`: required column missing
- `Input must have 'account_number' or 'account_number_id'`: ID column missing
- `No rows for obs_month = YYYY-MM`: month filter returned no data
- `pyarrow required for parquet...`: install pyarrow

