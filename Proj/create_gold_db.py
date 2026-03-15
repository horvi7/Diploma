import duckdb
import pandas as pd
import load_to_db as db

conn = duckdb.connect("database.duckdb")

global_warming_name = "cleaned_global_warming"
electric_vs_petrol_name = "cleaned_electric_vs_petrol_cars"
greenhouse_gas_name = "cleaned_greenhouse_gas_emissions"
sustainability_name = "cleaned_sustainability"

def get_cleaned_data(table_name):
    try:
        result = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        return result
    except Exception as e:
        print(f"Error fetching data from {table_name}: {e}")
        return None

global_warming_df = get_cleaned_data(global_warming_name)
electric_vs_petrol_df = get_cleaned_data(electric_vs_petrol_name)
greenhouse_gas_df = get_cleaned_data(greenhouse_gas_name)
sustainability_df = get_cleaned_data(sustainability_name)

co2_gold_df = pd.DataFrame()
if global_warming_df is not None:
    co2_gold_df = global_warming_df[["population", "gdp", "energy_consumption_per_capita", "urbanization", "industrial_activity", "renewable_energy_usage"]]

    db.load_to_db(co2_gold_df, "gold_co2_emissions")

    result = conn.execute("SELECT * FROM gold_co2_emissions").fetchdf()
    print("Gold CO2 Emissions Data:")
    print(result)