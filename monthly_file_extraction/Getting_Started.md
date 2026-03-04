# Getting Started

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Run

```bash
python test_file_automation.py healthplan_base_features.parquet 2026-02
```

Accepted month formats:

- `YYYY-MM` (example: `2026-02`)
- `MM-YYYY` (example: `02-2026`)

Optional output controls:

```bash
python test_file_automation.py healthplan_base_features.parquet 2026-02 --suffix v2 -o .\out
```

