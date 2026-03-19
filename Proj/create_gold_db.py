import duckdb
import pandas as pd
import db_handler as db

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

gold_co2_df = pd.DataFrame()
if global_warming_df is not None:
    gold_co2_df = global_warming_df[["co2_emissions","population", "gdp", "energy_consumption_per_capita", "urbanization", "industrial_activity", "renewable_energy_usage", "fossil_fuel_usage"]]

    db.load_to_db(gold_co2_df, "gold_co2_emissions")

    # result = conn.execute("SELECT * FROM gold_co2_emissions").fetchdf()
    # print("Gold CO2 Emissions Data:")
    # print(result)

gold_temperature_anomaly_df = pd.DataFrame()
if global_warming_df is not None:
    gold_temperature_anomaly_df = global_warming_df[["temperature_anomaly", "co2_emissions", "population", "gdp",
                                                     "energy_consumption_per_capita", "urbanization", "industrial_activity", "renewable_energy_usage", "fossil_fuel_usage", "deforestation_rate"]]

    
    db.load_to_db(gold_temperature_anomaly_df, "gold_temperature_anomalies")
    print("Gold Temperature Anomalies Data loaded into DuckDB:")
    # result = conn.execute("SELECT * FROM gold_temperature_anomalies").fetchdf()
    # print("Gold Temperature Anomalies Data:")
    # print(result)