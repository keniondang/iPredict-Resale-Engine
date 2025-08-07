import pandas as pd
from sqlalchemy import create_engine, text
import os
import urllib

def create_database_sql_server(data_dir='data'):
    """
    Reads all CSV files from a directory and loads them into a new SQL Server database.
    Each CSV file will become a table in the database.
    """
    # --- CONFIGURATION ---
    # !!! IMPORTANT: UPDATE THESE VALUES TO MATCH YOUR SQL SERVER SETUP !!!
    SERVER_NAME = "localhost\\SQLEXPRESS"  # e.g., "LAPTOP-12345\SQLEXPRESS" or your server's IP address
    DATABASE_NAME = "UsedPhoneResale" # The name of the empty database you created in SSMS.
    
    # This script uses Windows Authentication (trusted_connection=yes).
    # If you use SQL Server Authentication, you'll need to modify the connection string.
    # ---------------------

    try:
        # Create a connection string for SQL Server with Windows Authentication.
        # It uses the pyodbc driver, which is the standard for Python.
        params = urllib.parse.quote_plus(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={SERVER_NAME};"
            f"DATABASE={DATABASE_NAME};"
            f"Trusted_Connection=yes;"
        )
        connection_string = f"mssql+pyodbc:///?odbc_connect={params}"

        # Create a SQLAlchemy engine. The engine manages connections to the database.
        engine = create_engine(connection_string)

        # Test the connection.
        with engine.connect() as connection:
            print(f"Successfully connected to SQL Server: {SERVER_NAME}, Database: {DATABASE_NAME}")

            # List all the CSV files in the specified data directory.
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            
            if not csv_files:
                print(f"No CSV files found in the '{data_dir}' directory.")
                return

            print(f"\nFound {len(csv_files)} CSV files to load:")
            
            # Loop through each CSV file and load it into a new table.
            for file_name in csv_files:
                file_path = os.path.join(data_dir, file_name)
                
                # The table name will be the name of the file without the .csv extension.
                table_name = os.path.splitext(file_name)[0]
                
                print(f"  - Loading '{file_name}' into table '{table_name}'...")
                
                # Read the CSV file into a pandas DataFrame.
                df = pd.read_csv(file_path)
                
                # Write the DataFrame to the SQL Server database.
                # `if_exists='replace'` will drop the table first if it exists, then create it.
                # `index=False` prevents pandas from writing the DataFrame index as a column.
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
    # This block will run when you execute the script directly.
    # Make sure your CSV files are in a subdirectory named 'data'.
    create_database_sql_server()
