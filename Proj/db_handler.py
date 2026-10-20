import duckdb
import pandas as pd

import cleaning
import data_collector as dc

DB_PATH = "database.duckdb"

def load_to_db(df: pd.DataFrame, table_name: str) -> None:
    if not table_name.isidentifier():
        raise ValueError(f"Érvénytelen táblanév: '{table_name}'.")

    try:
        with duckdb.connect(DB_PATH) as conn:
            conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM df')
    except Exception as e:
        print(f"Hiba az adatok mentésekor. Tábla: {table_name} Ex: {e}")


def get_from_db(table_name: str) -> pd.DataFrame | None:
    if not table_name.isidentifier():
        raise ValueError(f"Érvénytelen táblanév: '{table_name}'.")

    try:
        with duckdb.connect(DB_PATH) as conn:
            return conn.table(table_name).to_df()
    except Exception as e:
        print(f"Hiba az adatok lekérdezésekor. Tábla: {table_name} Ex: {e}")
        return None

    
def refresh_database() -> None:
    """Load the latest collected datasets and their cleaned versions."""
    sources = dc.load_sources()
    for name in sources:
        raw_df = dc.get_latest_as_df(name)
        raw_name = f"raw_{name}"
        cleaned_name = f"cleaned_{name}"

        if raw_df is not None:
            load_to_db(raw_df, raw_name)
            cleaned_df = cleaning.clean_data(raw_df)
            if cleaned_df is not None:
                load_to_db(cleaned_df, cleaned_name)

