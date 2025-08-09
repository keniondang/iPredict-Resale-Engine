import pandas as pd
from sqlalchemy import create_engine, text
import os
import urllib

def create_database_sql_server(data_dir='data'):
    """
    Reads all CSV files from a directory and loads them into a new SQL Server database.
    Each CSV file will become a table in the database.
    """

    SERVER_NAME = "localhost\\SQLEXPRESS"  
    DATABASE_NAME = "UsedPhoneResale" 
    
    try:

        params = urllib.parse.quote_plus(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={SERVER_NAME};"
            f"DATABASE={DATABASE_NAME};"
            f"Trusted_Connection=yes;"
        )
        connection_string = f"mssql+pyodbc:///?odbc_connect={params}"

        engine = create_engine(connection_string)

        with engine.connect() as connection:
            print(f"Successfully connected to SQL Server: {SERVER_NAME}, Database: {DATABASE_NAME}")

            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            
            if not csv_files:
                print(f"No CSV files found in the '{data_dir}' directory.")
                return

            print(f"\nFound {len(csv_files)} CSV files to load:")
            
            for file_name in csv_files:
                file_path = os.path.join(data_dir, file_name)
                
                table_name = os.path.splitext(file_name)[0]
                
                print(f"  - Loading '{file_name}' into table '{table_name}'...")
                
                df = pd.read_csv(file_path)
                
                df.to_sql(table_name, engine, if_exists='replace', index=False)
                
                print(f"    ...Done. Table '{table_name}' created with {len(df)} rows.")

            print("\nDatabase creation and data loading complete.")
            print("You can now verify the tables in SQL Server Management Studio (SSMS).")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("\nPlease check the following:")
        print("1. Is the SQL Server instance running?")
        print("2. Is the SERVER_NAME and DATABASE_NAME in the script correct?")
        print("3. Have you installed the required libraries (`pip install sqlalchemy pyodbc`)?")
        print("4. Is the 'ODBC Driver 17 for SQL Server' installed?")

if __name__ == '__main__':
    create_database_sql_server()
