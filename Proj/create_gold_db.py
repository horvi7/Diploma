import duckdb
import pandas as pd
import db_handler as db
import numpy as np

conn = duckdb.connect("database.duckdb")
co2_name = "cleaned_co2"

def get_cleaned_data(table_name):
    try:
        result = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        return result
    except Exception as e:
        print(f"Error fetching data from {table_name}: {e}")
        return None

co2_df = get_cleaned_data(co2_name)

if co2_df is not None:
    gold_co2_df = co2_df[["co2", "gdp", "population", "energy_per_capita", "energy_per_gdp", "primary_energy_consumption"]].copy()
    gold_co2_df["log_energy_per_gdp"] = np.log1p(gold_co2_df["energy_per_gdp"])
    gold_co2_df["log_co2_per_energy"] = np.log1p(gold_co2_df["co2"] / gold_co2_df["energy_per_capita"])
    gold_co2_df["log_population"] = np.log1p(gold_co2_df["population"])
    gold_co2_df["log_gdp"] = np.log1p(gold_co2_df["gdp"])
    gold_co2_df["log_gdp_per_population"] = np.log1p(gold_co2_df["gdp"] / gold_co2_df["population"])

    gold_co2_df.drop(columns=["population", "gdp"], inplace=True)


co2_time_series_df = get_cleaned_data(co2_name)

if co2_time_series_df is not None:
    gold_co2_time_series_df = co2_time_series_df[["co2", "name", "year", "gdp", "population", "energy_per_capita", "energy_per_gdp", "primary_energy_consumption"]].copy()

    gold_co2_time_series_df["log_population"] = np.log1p(gold_co2_time_series_df["population"])
    gold_co2_time_series_df["log_gdp"] = np.log1p(gold_co2_time_series_df["gdp"])

    gold_co2_time_series_df.drop(columns=["population", "gdp"], inplace=True)

    db.load_to_db(gold_co2_df, "gold_co2")
    db.load_to_db(gold_co2_time_series_df, "gold_co2_time_series")