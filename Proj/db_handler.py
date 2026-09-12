import duckdb
import data_collector as dc
import pandas as pd
import cleaning

DB_PATH = "database.duckdb"

def load_to_db(df: pd.DataFrame, table_name: str) -> None:
    if not table_name.isidentifier():
        raise ValueError(f"Érvénytelen táblanév: '{table_name}'. "f"Csak betűket, számokat és aláhúzást tartalmazhat, és nem kezdődhet számmal.")

    try:
        with duckdb.connect(DB_PATH) as conn:
            conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM df')
    except Exception as e:
        print(f"Hiba az adatok mentésekor a(z) {table_name} táblába: {e}")


def get_from_db(table_name: str) -> pd.DataFrame | None:
    if not table_name.isidentifier():
        raise ValueError(f"Érvénytelen táblanév: '{table_name}'.")

    try:
        with duckdb.connect(DB_PATH) as conn:
            return conn.table(table_name).to_df()
    except Exception as e:
        print(f"Hiba az adatok lekérdezésekor a(z) {table_name} táblából: {e}")
        return None

    
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





