import pyodbc
import pandas as pd
import json
import os
import urllib
from sqlalchemy import create_engine

def main(config_path = "config.json"):
    """
    Fetches data from a SQL Server table and returns it as a DataFrame.
    
    Args:
        Config_path (str): Path to the configuration JSON file.
        
    Returns:
          pd.DataFramee: DataFrame containing the table data.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script Path: {script_dir}")
    config_file = os.path.join(script_dir, config_path)
    print(f"Config Path: {config_file}")

    # Load Configuration from JSON
    with open(config_file, "r") as file:
        config = json.load(file)

    # Read SQL Server connection details
    server = config["sql_server"]["server"]
    database = config["sql_server"]["database"]
    table = config["sql_server"]["table"]

    print(f"Server: {server}, Database: {database}, Table: {table}")

    # Define connections string for Windows Authetication
    connection_string = (
        f"DRIVER={{SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"Trusted_Connection=yes;"
    )
    print(f"{connection_string}")

    try:
        # Build a SQLAlchemy engine from the connection string
        # (pandas prefers this over a raw pyodbc connection)
        params = urllib.parse.quote_plus(connection_string)
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
        print("Connection to SQL Server Successful!")

        # Fetch data from the specified table
        query = f"SELECT * FROM {table}"
        df = pd.read_sql(query, engine)
        print(f"Data fetched Successfully form Table '{table}'..")
        return df
    except Exception as e:
        print(f"Error Connecting to SQL Server or Fetching data: {e}")
        return None


if __name__ == '__main__':
    df = main()
    if df is not None:
        print(f"\nFetched {len(df)} rows")
        print(df.head())