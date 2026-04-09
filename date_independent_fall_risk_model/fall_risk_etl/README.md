# Fall Risk ETL Pipeline

## Overview

This is an Extract, Transform, Load (ETL) pipeline designed to process and transform fall risk data. The pipeline extracts data from a SQL database, performs feature engineering including LLM-based sentiment analysis, aggregates data by month and account, and create parquet file ready to be ingested into the fall risk model script 

## Features

- **Database Connectivity**: Connects to SQL databases using ODBC
- **Multi-Customer Support**: Handles different customer types (health plans, direct-to-consumer, or all)
- **Date Range Filtering**: Flexibly filters data by month ranges
- **LLM Feature Extraction**: Integrates with LLM models to analyze operator notes for sentiment and intent detection
- **Feature Aggregation**: Aggregates multiple data sources by month and customer account
- **Parquet Export**: Exports processed data in efficient Parquet format for downstream analysis

## Prerequisites

### Dependencies

Install the required Python packages:

```bash
pip install pyodbc python-dotenv pandas matplotlib pyarrow
```

### Environment Variables

Create a `.env` file in the root directory with your database connection credentials:

```
DATABASE_USER=<your_username>
DATABASE_PASSWORD=<your_password>
DATABASE_SERVER=<your_server>
DATABASE_NAME=<your_database>
```

### Database Connection

The pipeline uses a `db_connection` module. To connect and query the Data warehouse, make sure you have the file.

### SQL Query Files

The script uses multiple SQL queries to pull the data, make sure to have the below SQL query files:
- `all_customers.sql` - Query for all customers
- `healthplan_customers.sql` - Query for health plan customers
- `dtc_customers.sql` - Query for direct-to-consumer customers
- `fall_alarms.sql` - Fall alarm events
- `buttons.sql` - Button press events
- `emergency_dispatch.sql` - Emergency dispatch events
- `steps.sql` - Step count data
- `operator_notes.sql` - Operator notes data

### LLM Sentiment Analysis

The pipeline uses a `Standard_Fall_Assist_Sentiment` module for processing operator notes. This module must implement:
- `process_file(file_path)` - Processes operator notes and returns sentiment features

## Usage

### Basic Usage

```python
from final_etl import FallRiskETL

# Initialize the ETL pipeline
etl = FallRiskETL(
    customer_type='health',           # or 'dtc' or 'all'
    training_start_date='2024-11',    # YYYY-MM format
    training_end_date='2025-04'       # YYYY-MM format
)

# Run the pipeline
final_df = etl.run_pipeline()

Executes the complete ETL pipeline:
1. Connects to database
2. Extracts all required tables
3. Filters to relevant customers and date range
4. Processes operator notes through LLM
5. Aggregates all features by month and account
6. Merges features into a comprehensive dataset
7. Returns final DataFrame


# Save the results
etl.save_data(final_df, '2024-11', '2025-04')

The pipeline generates a Parquet file with the following structure:

**Naming Convention**: `{YYYYMM_start}_to_{YYYYMM_end}_data.parquet`

**Location**: `data/health/` directory
```

### Configuration Parameters

When initializing `FallRiskETL`:

| Parameter             | Type | Description                                            | Default  |
|-----------------------|------|--------------------------------------------------------|----------|
| `customer_type`       | str  | Customer segment to process: 'health', 'dtc', or 'all' | Required |
| `training_start_date` | str  | Start date in YYYY-MM format                           | Required |
| `training_end_date`   | str  | End date in YYYY-MM format                             | Required |
| `prefix`              | str  | Path to SQL query files                                | Required |






**Key Columns**:
- `account_number` - Customer account identifier
- `account_id` - Internal account ID
- `age` - Customer age
- `health_plan` - Health plan name (acquisition_partner_name)
- `obs_month` - Observation period (Month)

**LLM Features**:
- `assist_count` - Count of assist flag occurrences
- `fall_count` - Count of fall flag occurrences
- `subscriber_reached_count` - Number of times subscriber was reached
- `help_sent_count` - Number of times help was sent
- `dispatch_cancelled_count` - Number of cancelled dispatches
- `sentiment_positive_count` - Count of positive sentiment notes
- `sentiment_negative_count` - Count of negative sentiment notes
- `sentiment_neutral_count` - Count of neutral sentiment notes

**Aggregated Features**:
- `fall_alarms_count` - Total fall alarms
- `buttons_count` - Total button presses
- `er_dispatch_count` - Total emergency dispatches
- `avg_daily_steps` - Average daily steps
- `prev_avg_daily_steps` - Previous month's average daily steps
- `steps_delta` - Change in average daily steps

## Data Flow

```
SQL Database
    ↓
Extract Tables (customers, alarms, buttons, etc.)
    ↓
Filter by Customer Type & Date Range
    ↓
LLM Sentiment Analysis (operator notes)
    ↓
Aggregate Features by Month/Account
    ↓
Merge All Features
    ↓
Export to Parquet
```

## Error Handling

The pipeline includes try-except blocks for data extraction. If any table fails to load, an exception is raised with details about which table failed.

## Performance Considerations

- Large date ranges may result in significant data volumes
- LLM processing can be time-intensive depending on the volume of operator notes
- Parquet format provides efficient compression and columnar storage
- Filter by customer type and date range to minimize unnecessary data processing

## Troubleshooting

### Database Connection Issues
- Verify `.env` file contains correct credentials
- Check database server is accessible and running
- Ensure ODBC drivers are installed

### Missing SQL Files
- Verify all `.sql` files exist in the configured `prefix` directory
- Check SQL syntax in query files

### LLM Processing Errors
- Ensure `Standard_Fall_Assist_Sentiment` module is properly installed
- Check operator notes file format is compatible with LLM processor

### Memory Issues with Large Datasets
- Reduce the date range
- Process by customer type separately
- Increase system RAM or use data chunking approaches

## Example Output

```
Connecting to the database...
Extracting data from the database...
Count of Active Customers in this training range for health customers: 15,234
Observation date range for training: 2024-11-01 to 2025-04-30
Aggregating features by month and account number
Creating a record for each account number for each month in the observation period
Merging LLM features with customers
Saved successfully to: data/health/202411_to_202504_data.parquet
Final dataframe shape: (304,680)
.......
