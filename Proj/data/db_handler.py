import os

import duckdb
import pandas as pd
import numpy as np

from data import cleaning
from data import data_collector as dc

DB_PATH = os.environ.get("DATABASE_PATH", "database.duckdb")

def load_to_db(df: pd.DataFrame, table_name: str) -> None:
    if not table_name.isidentifier():
        raise ValueError(f"Érvénytelen táblanév: '{table_name}'.")

    try:
        with duckdb.connect(DB_PATH) as conn:
            conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            conn.from_df(df).create(table_name)
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

def get_cleaned_data(table_name: str) -> pd.DataFrame | None:
    try:
        return get_from_db(table_name)
    except Exception as e:
        print(f"Error fetching data from {table_name}: {e}")
        return None

def create_gold_tables() -> None:
    co2_df = get_cleaned_data("cleaned_co2")

    if co2_df is None:
        return

    gold_co2_df = co2_df[[
        "co2", "gdp", "population", "energy_per_capita", "energy_per_gdp",
        "primary_energy_consumption"
    ]].copy()
    gold_co2_df["energy_per_gdp"] = np.log1p(gold_co2_df["energy_per_gdp"])

    gold_co2_df["gdp_per_population"] = np.log1p(gold_co2_df["gdp"] / gold_co2_df["population"])
    gold_co2_df["population"] = np.log1p(gold_co2_df["population"])
    gold_co2_df["gdp"] = np.log1p(gold_co2_df["gdp"])

    gold_co2_time_series_df = co2_df[[
        "co2", "name", "year", "gdp", "population", "energy_per_capita",
        "energy_per_gdp", "primary_energy_consumption"
    ]].copy()
    gold_co2_time_series_df["population"] = np.log1p(gold_co2_time_series_df["population"])
    gold_co2_time_series_df["gdp"] = np.log1p(gold_co2_time_series_df["gdp"])

    load_to_db(gold_co2_df, "gold_co2")
    load_to_db(gold_co2_time_series_df, "gold_co2_time_series")

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
                create_gold_tables()