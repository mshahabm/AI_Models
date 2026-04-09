
import pyodbc
import os, glob, re
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
from db_connection import get_connection_string,execute_query, test_connection  
import calendar
from typing import Tuple
import Standard_Fall_Assist_Sentiment as sentiment


class FallRiskETL:
    def __init__(self , customer_type:str, prefix:str,training_start_date:str, training_end_date:str = None ):
        self.prefix = prefix
        self.customer_type = customer_type
        self.training_start_date = training_start_date
        self.training_end_date = training_end_date
        

        # custmer types
        self.ALL_CUSTOMER_QUERY = f'{self.prefix}/all_customers.sql'
        self.HEALTH_CUSTOMER_QUERY = f'{self.prefix}/healthplan_customers.sql'
        self.DTC_CUSTOMER_QUERY = f'{self.prefix}/dtc_customers.sql'

        # base features
        self.FALL_ALARMS_QUERY = f'{self.prefix}/fall_alarms.sql'
        self.BUTTONS_QUERY = f'{self.prefix}/buttons.sql'
        self.EMERGENCY_QUERY = f'{self.prefix}/emergency_dispatch.sql'
        self.STEPS_QUERY = f'{self.prefix}/steps.sql'
        self.OPERATOR_NOTES_QUERY = f'{self.prefix}/operator_notes.sql'

    # function to connect to the database using the connection string from the .env file.

    def connect_to_db(self) -> pyodbc.Connection:
        """Connect to the database using the connection string from the .env file."""
        print("Connecting to the database...")
        connection_string = get_connection_string('datahub')
        test_connection(connection_string)
        return connection_string

    # Function to pull data from databwarehouse for a given table name and return as a dataframe.

    def get_data(self, table_name: str, connection_string: str) -> pd.DataFrame:
        query_path = f'{self.prefix}\{table_name}.sql'
        df = execute_query(query_path, connection_string)
        return df
    
    def inc_ym(ym: str) -> str:
        yyyy, mm = map(int, ym.split("-"))
        mm += 1
        if mm > 12:
            mm = 1
            yyyy += 1
        return f"{yyyy}-{mm:02d}"


    # Function to find the most recent training file in the current directory based on the naming convention "FallRisk_Training_MMYYYY_To_MMYYYY_Healthplans.parquet". It extracts the end date from the filename to determine which file is the latest. If no files are found or if the dates cannot be parsed, it raises an appropriate error.
    def find_latest_training_file(self, customer_type:str) -> str:
        """Find the most recent training file"""
        cur_dir = os.getcwd()
        new_path = os.path.join(cur_dir, 'data',{customer_type})
        os.chdir(new_path)
        files = glob.glob(f'*_To_*_{customer_type}.*')
        
        if not files:
            raise FileNotFoundError("No training files found")
        
        file_dates = []
        for file in files:
            match = re.search(r'_To_(\d{2})(\d{4})_{customer_type}', file)
            if match:
                mm, yyyy = int(match.group(1)), int(match.group(2))
                file_dates.append((yyyy * 100 + mm, f'{yyyy}-{mm:02d}'.format(yyyy, mm)))
        
        if not file_dates:
            raise ValueError("Cannot parse dates from training filenames")
        
        file_dates.sort(reverse=True)
        latest_date = file_dates[0][1]

        latest_date = self.inc_ym(latest_date)
        
        print(f"[OK] Auto-detected: {latest_date} as the next training month based on existing files.")
        return latest_date

    # funciton to get the full data rangefor a given start and end month, ensuring that the end date includes all days in the end month.

    def get_month_range(self,  start_date: str, end_date: str = None) -> Tuple[str, str]:
        """Get full range between start and (optionally) end month, including all days in end month.
        If end_date not given, defaults to start_date."""
        if end_date is None or not end_date.strip():
            end_date = start_date
        start_full = f"{start_date}-01"
        end_year, end_month = map(int, end_date.split('-'))
        last_day = calendar.monthrange(end_year, end_month)[1]
        end_full = f"{end_date}-{last_day:02d}" # 02d to pad as 2 digits always
        start = pd.to_datetime(start_full).strftime('%Y-%m-%d')
        end = pd.to_datetime(end_full).strftime('%Y-%m-%d')
        return start, end

    # function to filter dataframe by account numbers and date range.

    def filter_by_date_and_account(self, df: pd.DataFrame,col:str, count_of_customers:list, OBS_START_DATE:str, OBS_END_DATE:str) -> pd.DataFrame:
        """Filter dataframe by account numbers and date range."""
        df[col] = pd.to_datetime(df[col])
        filtered_df = df[
            (df['account_number'].isin(count_of_customers)) & 
            (df[col] >= OBS_START_DATE) &
            (df[col] <= OBS_END_DATE)
        ].sort_values('account_number')
        return filtered_df

    # function to save operator notes subset for given customer type and date range.

    def save_operator_notes_subset(self, operator_notes_df: pd.DataFrame, customer_type: str, obs_start_date: str, obs_end_date: str):
        start_yyyymm = pd.to_datetime(obs_start_date).strftime('%Y%m')
        end_yyyymm = pd.to_datetime(obs_end_date).strftime('%Y%m')
        file_name = f"{start_yyyymm}_to_{end_yyyymm}_operator.parquet"

        output_dir = os.path.join(os.getcwd(), 'data', customer_type)
        os.makedirs(output_dir, exist_ok=True)

        parquet_path = os.path.join(output_dir, file_name)

        operator_notes_df.to_parquet(parquet_path)
        print(f"Saved operator notes for '{customer_type}' to: {parquet_path}")

        return parquet_path

    # function to handle LLM features, converting categorical to numeric and preparing for aggregation by month and account number, need to improve the funtion.

    def prepare_llm_feature_for_aggregation(self, llm_feature_df: pd.DataFrame, df: pd.DataFrame ) -> pd.DataFrame:
        """ convert categorical features to numeric and prepare for aggregation."""
        llm_features_df_1 = llm_feature_df[[
            'account_number',
            'alarm_date',
            'fall_flag',
            'assist_flag',
            'sentiment_positive_flag',
            'sentiment_negative_flag',
            'sentiment_neutral_flag',
            'subscriber_reached',
            'help_sent',
            'dispatch_cancelled'
                ]]
            
        for column in llm_features_df_1.columns:
            if column in ['subscriber_reached', 'help_sent', 'dispatch_cancelled']:
                llm_features_df_1[column] = llm_features_df_1[column].map({'Yes': 1, 'No': 0})
        
        llm_features_w_customers = df.merge(
            llm_features_df_1,
            on = 'account_number',
            how = 'left'
        )
        llm_features_w_customers['obs_month'] = pd.to_datetime(llm_features_w_customers['alarm_date']).dt.to_period('M')

        llm_features_agg = (
            llm_features_w_customers.groupby(['account_number','account_id','account_record_type_name', 'obs_month'])
            .agg(
                sentiment_positive_count = ('sentiment_positive_flag', 'sum'),
                sentiment_negative_count = ('sentiment_negative_flag', 'sum'),
                sentiment_neutral_count = ('sentiment_neutral_flag', 'sum'),
                assist_flag_count = ('assist_flag', 'sum'),
                fall_flag_count = ('fall_flag', 'sum'),
                subscriber_reached_count = ('subscriber_reached', 'sum'), 
                help_sent_count = ('help_sent', 'sum'),
                dispatch_cancelled_count = ('dispatch_cancelled', 'sum')
            )
            .reset_index()
            ).sort_values(by = ['account_number', 'obs_month'], ascending = True)
        

        for column in llm_features_agg.columns:
            if column not in ['account_number','account_id', 'account_record_type_name', 'obs_month']:
                llm_features_agg[column] = llm_features_agg[column].astype(int)

        
        return llm_features_agg

    ## code to aggregate features by month and account number, summing up all feature counts for each month-account combination.
    def aggregate_features_by_month_and_account_number(self,df: pd.DataFrame, df_name:str, col_name:str , col:str) -> pd.DataFrame:
        """ Aggregate features by month and account number, summing up all feature counts for each month-account combination."""
        if df_name == 'steps_df':
            df['obs_month'] = pd.to_datetime(df['step_date']).dt.to_period('M')
            df_agg = (
                df
                .groupby(['account_number', 'obs_month'])
                .agg(
                    avg_daily_steps = ('daily_steps', lambda x: round(x.mean(), 2)),
                    prev_avg_daily_steps = ('daily_steps', lambda x: round(x.shift(1).mean(), 2)),
                    steps_delta = ('daily_steps', lambda x: round(x.diff().mean(), 2))
                )
                .reset_index())
            return df_agg

        else:    
            df['obs_month'] = pd.to_datetime(df[col]).dt.to_period('M')
            df_agg = (
                df
                .groupby(['account_number', 'obs_month'])
                .size()
                .reset_index(name = col_name))
            return df_agg
        
    ## code to perform left joins
    def perform_left_join(self, df:pd.DataFrame, df_to_join:pd.DataFrame) -> pd.DataFrame:
        """Perform left join on account number and obs_month."""
        merged_df = df.merge(
            df_to_join,
            on = ['account_number', 'obs_month'],
            how = 'left'
        )
        return merged_df

    def save_data(self, df: pd.DataFrame, customer_type: str, obs_start_date: str, obs_end_date: str):
        start_yyyymm = pd.to_datetime(obs_start_date).strftime('%Y%m')
        end_yyyymm = start_yyyymm if obs_end_date == None else pd.to_datetime(obs_end_date).strftime('%Y%m')
        file_name = f"{start_yyyymm}_to_{end_yyyymm}_{customer_type}.parquet"
        output_dir = os.path.join(os.getcwd(), 'data', customer_type)
        os.makedirs(output_dir, exist_ok=True)
        parquet_path = os.path.join(output_dir, file_name)
        df.to_parquet(parquet_path)

    def get_col_name(self,training_start_date:str, training_end_date:str) -> str:
        if training_end_date is None:
            return f'{training_start_date[-2:]}_{training_start_date[:4]}'
        else:
            return f'{training_end_date[-2:]}_{training_end_date[:4]}'


    def run_pipeline(self):

        # connect to the database
        connection_string = self.connect_to_db()

        # get the data for the specified customer type and date range
        print("Extracting data from the database...")


        tables = ['buttons', 'fall_alarms', 'steps', 'emergency_dispatch', 'operator_notes', 'healthplan_customers', 'dtc_customers', 'all_customers']
        
        df_table = {}
        for table in tables:
            try:
                df_table[f'{table}_df'] = self.get_data(table, connection_string)
            except Exception as e:
                print(f"Error fetching data for {table}: {e}")
                raise
        if self.customer_type == 'health':
            df = df_table['healthplan_customers_df'].copy()
        elif self.customer_type == 'dtc':
            df = df_table['dtc_customers_df'].copy()
        elif self.customer_type == 'all':
            df =  df_table['all_customers_df'].copy()
        else:
            raise ValueError(f"Invalid CUSTOMER_TYPE: {self.customer_type}. Should be 'health', 'dtc', or 'all'")
        

        # Ectract month range for training and filter operator notes by date and account number, then send the file to LLM for feature extraction and prepare llm features .
        count_of_customers = df['account_number'].unique()
        print(f"Count of Active Customers in this training range for {self.customer_type} customers: {len(count_of_customers):,}")
        OBS_START_DATE, OBS_END_DATE = self.get_month_range(self.training_start_date, self.training_end_date)
        print(f"Observation date range for training: {OBS_START_DATE} to {OBS_END_DATE}")
        operator_notes_df = self.filter_by_date_and_account(df_table['operator_notes_df'],'alarm_date', count_of_customers, OBS_START_DATE, OBS_END_DATE)
        saved_file_path = self.save_operator_notes_subset(operator_notes_df, self.customer_type, OBS_START_DATE, OBS_END_DATE)

        ## sending to llm
        llm_features = sentiment.process_file(saved_file_path)

        ## preparing llm features for aggregation
        llm_features_agg = self.prepare_llm_feature_for_aggregation(llm_features, df)
        print(llm_features_agg.dtypes)


        # Prepare base features by filtering for the relevant date range and account numbers, then aggregate by month and account number to get monthly counts for each feature.
        agg_table = ['fall_alarms_df', 'buttons_df', 'emergency_dispatch_df', 'steps_df']
        col_name = ['alarm_date','button_press_date','dispatch_date','step_date']
        df_agg_tables ={}

        print("Aggregating features by month and account number")
        for table, col in zip(agg_table, col_name):
            col_name = table.replace('_df', '') + '_count'
            filter_tble = self.filter_by_date_and_account(df_table[table], col,count_of_customers, OBS_START_DATE, OBS_END_DATE)
            df_agg_tables[f'{table}_agg'] = self.aggregate_features_by_month_and_account_number(filter_tble, table, col_name,col)
            print(f"Aggregated {table} features by month and account number")
        
        # Create a record for each account number for each month in the obs
        print("Creating a record for each account number for each month in the observation period")
        all_months = pd.period_range(OBS_START_DATE, OBS_END_DATE, freq='M')
        accounts = df[['account_number']].drop_duplicates().reset_index(drop=True)
        accounts['key'] = 1
        months_df = pd.DataFrame({'obs_month': all_months})
        months_df['key'] = 1
        account_months = accounts.merge(months_df, on='key').drop('key', axis=1)
        customer_expanded = account_months.merge(df, on='account_number', how='left')


        # Merge LLM features with customers, then iteratively merge each aggregated feature table with the customers, ensuring that we maintain all customer records and add the relevant features for each month-account combination. Finally, reorder columns to have key identifiers first, followed by LLM features, then aggregated features, and finally any remaining columns.
        print("Merged customers with expanded account-month combinations")
        print("Merging LLM features with customers")
        customers_w_llm = customer_expanded.merge(llm_features_agg[
                                                                    ['account_number',
                                                                    'obs_month',
                                                                    'assist_flag_count', 
                                                                    'fall_flag_count', 
                                                                    'subscriber_reached_count', 
                                                                    'help_sent_count', 
                                                                    'dispatch_cancelled_count', 
                                                                    'sentiment_positive_count', 
                                                                    'sentiment_negative_count', 
                                                                    'sentiment_neutral_count'
                                                                    ]]
                                                  , on = ['account_number', 'obs_month'], how = 'left')
        print("Merging aggregated features with customers iteratively")
        customer_w_tables = {}
        stack = [customers_w_llm]
        i = 0
        while stack and i < 4:
            table = agg_table[i]
            customer_w_tables[f'customers_w_{table}'] = self.perform_left_join(stack[-1], df_agg_tables[f'{table}_agg'])
            stack.append(customer_w_tables[f'customers_w_{table}'])
            print(f"Merged {table}_agg with customers")
            i += 1

        # Prepare final dataset for saving by renaming columns for clarity and reordering to have key identifiers first, followed by LLM features, then aggregated features, and finally any remaining columns.       
        final = customer_w_tables['customers_w_steps_df'].rename(columns = {'account_record_type_name': 'brand', 'acquisition_partner_name': 'health_plan','emergency_dispatch_count': 'er_dispatch_count', 'assist_flag_count': 'assist_count','fall_flag_count': 'fall_count'}).copy()
        cols = final.columns.tolist()
        print("Reordering columns to have key identifiers first, followed by LLM features, then aggregated features, and finally any remaining columns.")
        health_desired_order = ['account_number', 'account_id', 'age', 'health_plan', 'obs_month','brand']

        # desired_order = ['account_number', 'account_id', 'age', 'brand', 'obs_month']
        existing_order = [col for col in health_desired_order if col in cols]
        rest = [col for col in cols if col not in existing_order]
        final = final[existing_order + rest]
        col_name = self.get_col_name(self.training_start_date, self.training_end_date)
        if self.customer_type == 'dtc':
            final.drop(columns = ['account_id','obs_month'], inplace = True)
        if self.training_end_date is None:
            for col in rest:
                final.rename(columns = {col: f'{col}_{col_name}'}, inplace = True)

        return final
    
if __name__ == "__main__":
    # Define parameters
    CUSTOMER_TYPE = 'health'  # 'health', 'dtc', or 'all'
    TRAINING_START_DATE = '2025-01' # Format: 'YYYY-MM'
    TRAINING_END_DATE = '2026-03' # Format: 'YYYY-MM' or None for single month training
  
    PREFIX = r'C:\Users\ShahabMemon\Documents\Work\fall_risk_model\fall_risk_etl\sql'

    etl = FallRiskETL( CUSTOMER_TYPE,  PREFIX,TRAINING_START_DATE, TRAINING_END_DATE)
    final_df = etl.run_pipeline()

    print("Final dataframe shape:", final_df.shape)
    print("Final dataframe columns:", final_df.columns)
    etl.save_data(final_df, CUSTOMER_TYPE, TRAINING_START_DATE, TRAINING_END_DATE)

# Column summaries for the 'final' DataFrame:
# account_number: Unique identifier for the customer/account.
# account_id: Another unique account identifier, possibly from Salesforce or CRM system.
# age: Age of the customer.
# brand: The name/type of account record, e.g., Medicaid Customer (renamed from account_record_type_name).
# health_plan: The associated health plan or acquisition partner (renamed from acquisition_partner_name).
# obs_month: Observation month represented as YYYY-MM.
# sentiment_positive_count: Number of positive sentiment notes in the month.
# sentiment_negative_count: Number of negative sentiment notes in the month.
# sentiment_neutral_count: Number of neutral sentiment notes in the month.
# assist_flag_count: Count of Assist alarms in the month. ----- change the name from assistt_count to assist_flag_count
# all_flag_count: Count of Fall alarms in the month.   --- not in the table
# subscriber_reached_count: Number of times the subscriber was reached in the month.
# help_sent_count: Number of alarms where help was sent in the month.
# dispatch_cancelled_count: Number of times a dispatch was cancelled in the month.
# fall_alarm_count: Total number of fall alarms (from alarms data). --- change the name to fall_alarm_count from fall_count
# button_press_count: Total number of button presses during the month. -- not in the table
# emergency_dispatch_count: Count of ER/emergency dispatches for the month. --- not in the table
# avg_daily_steps: Average daily steps for the observation month.
# prev_avg_daily_steps: Average daily steps for the previous month.
# steps_delta: Difference in average daily steps from previous to current month.

## assist_count: assist_flag_count, fall_count: all_flag_count 


