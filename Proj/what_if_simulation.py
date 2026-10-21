import numpy as np

import db_handler as db
import trained_models as tm


def predict_and_visualize():
    co2_df = db.get_from_db("gold_co2")
    row = co2_df.iloc[0]

    # The gold table stores these Kaya factors as log1p-transformed values.
    population = np.expm1(row["log_population"])
    gdp_per_population = np.expm1(row["log_gdp_per_population"])
    energy_per_gdp = np.expm1(row["log_energy_per_gdp"])

    # CO2 intensity is derived from the observed target and is not used as a
    # machine-learning feature.
    energy = population * gdp_per_population * energy_per_gdp
    co2_per_energy = row["co2"] / energy

    changed_population = population * 0.9
    original_features = co2_df[
        ["log_population", "log_gdp_per_population", "log_energy_per_gdp"]
    ].iloc[[0]]
    changed_features = original_features.copy()
    changed_features["log_population"] = np.log1p(changed_population)

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
        model = tm.get_model(model_name)
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
