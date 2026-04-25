import duckdb
import data_collector as dc
import pandas as pd
import cleaning

def load_to_db(df, table_name):
    conn = duckdb.connect("database.duckdb")

    conn.register("temp_df", df)

    conn.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * FROM temp_df""")

    conn.close()

sources = dc.load_sources()
for name, url in sources.items():
    raw_df = dc.get_latest_as_df(name)
    cleaned_df = cleaning.clean_data(raw_df)
    raw_name = f"raw_{name}"
    cleaned_name = f"cleaned_{name}"

    if raw_df is not None:
        load_to_db(raw_df, raw_name)
    if cleaned_df is not None:
        load_to_db(cleaned_df, cleaned_name)


def get_from_db(table_name: str) -> pd.DataFrame:
    conn = duckdb.connect("database.duckdb")
    try:
        result = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        return result
    except Exception as e:
        print(f"Error fetching data from {table_name}: {e}")
        return None
    finally:
        conn.close()
