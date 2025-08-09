import pandas as pd
from sqlalchemy import create_engine
import urllib

SERVER_NAME = "localhost\\SQLEXPRESS"
DATABASE_NAME = "UsedPhoneResale"

def get_db_engine():
    """
    Creates and returns a SQLAlchemy engine for the SQL Server database.
    Handles connection string creation and provides error handling.
    """
    try:
        params = urllib.parse.quote_plus(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={SERVER_NAME};"
            f"DATABASE={DATABASE_NAME};"
            f"Trusted_Connection=yes;"
        )
        connection_string = f"mssql+pyodbc:///?odbc_connect={params}"
        engine = create_engine(connection_string)
        connection = engine.connect()
        connection.close()
        print(f"Successfully created DB engine for SQL Server: {SERVER_NAME}, DB: {DATABASE_NAME}")
        return engine
    except Exception as e:
        print(f"FATAL: Could not create database engine. Error: {e}")
        return None

def load_data_from_db(engine, table_name):
    """
    Loads data from a specified table into a pandas DataFrame.
    """
    try:
        print(f"  - Loading table: '{table_name}'...")
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
        print(f"    ...Done. Loaded {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error loading table {table_name}: {e}")
        return pd.DataFrame()