import numpy as np

import db_handler as db
import trained_models as tm


def predict_and_visualize():
    co2_df = db.get_from_db("gold_co2")
    co2_df = co2_df[[
        "co2",
        "log_population",
        "log_gdp_per_population",
        "log_energy_per_gdp",
        "log_co2_per_energy",
    ]]

    features = co2_df[[
        "log_population",
        "log_gdp_per_population",
        "log_energy_per_gdp",
        "log_co2_per_energy",
    ]]

    test_data = features.iloc[[0]]

    changed_test_data = test_data.copy()
    changed_test_data["log_population"] = changed_test_data["log_population"] + np.log(0.9)

    row = test_data.iloc[0]
    changed_row = changed_test_data.iloc[0]

    kaya_log_orig = (
        row["log_population"]
        + row["log_gdp_per_population"]
        + row["log_energy_per_gdp"]
        + row["log_co2_per_energy"]
    )
    kaya_log_changed = (
        changed_row["log_population"]
        + changed_row["log_gdp_per_population"]
        + changed_row["log_energy_per_gdp"]
        + changed_row["log_co2_per_energy"]
    )

    # Visszaalakítás a tényleges értékre
    kaya_orig_val = np.exp(kaya_log_orig)
    kaya_changed_val = np.exp(kaya_log_changed)
    kaya_pct = ((kaya_changed_val / kaya_orig_val) - 1) * 100

    print(f"Kaya Eredeti kibocsátás: {kaya_orig_val:.2f}")
    print(f"Kaya Változott kibocsátás: {kaya_changed_val:.2f}")
    print(f"Kaya tényleges növekedése: +{kaya_pct:.2f}%\n")

    # Modell predikció
    random_forest_model = tm.get_model('RandomForest')

    lightgbm_model = tm.get_model('LightGBM')

    xgboost_model = tm.get_model('XGBoost')

    #Itt majd Írás előtt Kaya Identity scenariokat kipróbálni és a modelleket öszehasonlítani
    #Most csak random próba

    pred_orig_random_forest = random_forest_model.predict(test_data)[0]
    pred_orig_lightgbm = lightgbm_model.predict(test_data)[0]
    pred_orig_xgboost = xgboost_model.predict(test_data)[0]

    pred_changed_random_forest = random_forest_model.predict(changed_test_data)[0]
    pred_changed_lightgbm = lightgbm_model.predict(changed_test_data)[0]
    pred_changed_xgboost = xgboost_model.predict(changed_test_data)[0]

    model_pct = ((pred_changed_random_forest / pred_orig_random_forest) - 1) * 100
    print(f"Random Forest Predikció (Eredeti): {pred_orig_random_forest:.2f}")
    print(f"Random Forest Predikció (Változott): {pred_changed_random_forest:.2f}")
    print(f"Modell által becsült változás: +{model_pct:.2f}%")

    print(f"LightGBM Predikció (Eredeti): {pred_orig_lightgbm:.2f}")
    print(f"LightGBM Predikció (Változott): {pred_changed_lightgbm:.2f}")
    model_pct = ((pred_changed_lightgbm / pred_orig_lightgbm) - 1) * 100
    print(f"Modell által becsült változás: +{model_pct:.2f}%")

    print(f"XGBoost Predikció (Eredeti): {pred_orig_xgboost:.2f}")
    print(f"XGBoost Predikció (Változott): {pred_changed_xgboost:.2f}")
    model_pct = ((pred_changed_xgboost / pred_orig_xgboost) - 1) * 100
    print(f"Modell által becsült változás: +{model_pct:.2f}%")

predict_and_visualize()