import pyodbc
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv('env_sqlserver.env')

def get_driver():
    drivers = [d for d in pyodbc.drivers() if d.startswith('ODBC Driver')]
    if not drivers:
        raise Exception("No ODBC Driver found. Please install an ODBC driver for SQL Server.")
    return drivers[0]

def get_connection_string(database_name):
    driver = get_driver()
    server = os.getenv('SQL_SERVER')
    username = os.getenv('SQL_USERNAME')
    password = os.getenv('SQL_PASSWORD')
    if not (server and username and password):
        raise ValueError("Missing required environment variables. Check env_sqlserver.env file.")
    return (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database_name};"
        f"UID={username};"
        f"PWD={password}"
    )

def test_connection(connection_string=None):
    if not connection_string:
        print("No connection string provided")
        return False
    try:
        with pyodbc.connect(connection_string):
            print("Successfully connected to SQL Server")
    except Exception as e:
        print(f"Error connecting to SQL Server: {e}")
        return None

def execute_query(query_path, connection_string=None):
    if not connection_string:
        print("No connection string provided")
        return None
    try:
        with open(query_path, 'r', encoding='utf-8') as file:
            query = file.read()
    except Exception as e:
        print(f"Error reading query file '{query_path}': {e}")
        return None
    try:
        with pyodbc.connect(connection_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                return pd.DataFrame.from_records(data, columns=columns)
    except Exception as e:
        print(f"Error executing query: {e}")
        return None
