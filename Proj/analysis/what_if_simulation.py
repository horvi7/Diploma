import numpy as np
import pandas as pd
from data import db_handler as db
from models import trained_models as tm


def predict_and_visualize(df: pd.DataFrame):
    row = df.iloc[0]

    # Alapból logaritmikus skálán vannak az adatok, ezért vissza kell alakítani őket
    population = np.expm1(row["population"])
    gdp_per_population = np.expm1(row["gdp_per_population"])
    energy_per_gdp = np.expm1(row["energy_per_gdp"])

    # CO2 intenstiy target leakage lenne
    energy = population * gdp_per_population * energy_per_gdp
    co2_per_energy = row["co2"] / energy

    changed_population = population * 0.9
    original_features = df[["population", "gdp_per_population", "energy_per_gdp"]].iloc[[0]]
    changed_features = original_features.copy()
    changed_features["population"] = np.log1p(changed_population)

    original_co2 = (
        population
        * gdp_per_population
        * energy_per_gdp
        * co2_per_energy
    )
    changed_co2 = (
        changed_population
        * gdp_per_population
        * energy_per_gdp
        * co2_per_energy
    )
    co2_change = (changed_co2 / original_co2 - 1) * 100

    print(f"Eredeti népesség: {population:.2f}")
    print(f"Változott népesség: {changed_population:.2f}")
    print(f"Eredeti CO2: {original_co2:.2f}")
    print(f"Változott CO2: {changed_co2:.2f}")
    print(f"Kaya Identity CO2 változása: {co2_change:.2f}%\n")

    for model_name in ("RandomForest", "LightGBM", "XGBoost"):
        model = tm.get_model(model_name, df[["population", "gdp_per_population", "energy_per_gdp"]], df["co2"])
        original_prediction = model.predict(original_features)[0]
        changed_prediction = model.predict(changed_features)[0]
        model_change = (changed_prediction / original_prediction - 1) * 100
        difference = model_change - co2_change

        print(f"{model_name}:")
        print(f"  Eredeti predikció: {original_prediction:.2f}")
        print(f"  Változott predikció: {changed_prediction:.2f}")
        print(f"  Modell CO2 változása: {model_change:.2f}%")
        print(f"  Eltérés a Kaya-változástól: {difference:.2f} százalékpont\n")


if __name__ == "__main__":
    predict_and_visualize()
