from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras, reshape
import train_models_with_optuna as train_optuna
import db_handler as db
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, activations
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


rfr_params = {'n_estimators': 105, 'max_depth': 6, 'min_samples_split': 9, 'min_samples_leaf': 4, 'bootstrap': True, 'max_leaf_nodes': 24}
lgbm_params = {'n_estimators': 167, 'learning_rate': 0.011882888130545924, 'max_depth': 4, 'min_child_samples': 23, 'subsample': 0.9390586696763323, 'colsample_bytree': 0.6506930125803669, 'reg_alpha': 0.9821512272350051, 'reg_lambda': 0.11827601292607956}
xgb_params = {'n_estimators': 381, 'learning_rate': 0.11854834150995305, 'max_depth': 6, 'min_child_weight': 5, 'gamma': 4.4277220131081, 'subsample': 0.9850997649876527, 'colsample_bytree': 0.6318153394672948, 'reg_alpha': 0.4635012413919827, 'reg_lambda': 0.10254642573937399}

def predict_and_visualize():
    co2_df = db.get_from_db("gold_co2")
    co2_df = co2_df[[
        "co2",
        "log_population",
        "log_gdp_per_population",
        "log_energy_per_gdp",
        "log_co2_per_energy",
    ]]

    target = co2_df["co2"]
    features = co2_df[[
        "log_population",
        "log_gdp_per_population",
        "log_energy_per_gdp",
        "log_co2_per_energy",
    ]]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=32
    )

    # Eredeti tesztsor
    test_data = pd.DataFrame([{
        "log_population": 16.656126,
        "log_gdp_per_population": 10.207357,
        "log_energy_per_gdp": 2.265,
        "log_co2_per_energy": 0.004512,
    }])

    # 1. HELYES 10%-os növelés a log-térben: ln(1.1)-et ADUNK HOZZÁ
    changed_test_data = test_data.copy()
    changed_test_data["log_population"] = changed_test_data["log_population"] + np.log(0.9)

    # 2. Kaya-identitás logaritmikus alakban (ÖSSZEADÁS)
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

    # Visszaalakítás a tényleges fizikai skálára (exp):
    kaya_orig_val = np.exp(kaya_log_orig)
    kaya_changed_val = np.exp(kaya_log_changed)
    kaya_pct = ((kaya_changed_val / kaya_orig_val) - 1) * 100

    print(f"Kaya Eredeti kibocsátás: {kaya_orig_val:.2f}")
    print(f"Kaya Változott kibocsátás: {kaya_changed_val:.2f}")
    print(f"Kaya tényleges növekedése: +{kaya_pct:.2f}%\n")

    # Modell predikció
    random_forest_model = RandomForestRegressor(**rfr_params, random_state=32)
    random_forest_model.fit(x_train, y_train)

    lightgbm_model = lgb.LGBMRegressor(**lgbm_params, random_state=32)
    lightgbm_model.fit(x_train, y_train)

    xgboost_model = xgb.XGBRegressor(**xgb_params, random_state=32, n_jobs=-1)
    xgboost_model.fit(x_train, y_train)

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