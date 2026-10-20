import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split

import db_handler as db
from model_parameters import LGBM_PARAMS, RFR_PARAMS, XGB_PARAMS

def get_model(model_type):
    co2_df = db.get_from_db("gold_co2")
    print(co2_df.head())
    #Ez Kaya
    # co2_df = co2_df[[
    #     "co2",
    #     "log_population",
    #     "log_gdp_per_population",
    #     "log_energy_per_gdp",
    #     "log_co2_per_energy",
    # ]]

    # target = co2_df["co2"]
    # features = co2_df[[
    #     "log_population",
    #     "log_gdp_per_population",
    #     "log_energy_per_gdp",
    #     "log_co2_per_energy",
    # ]]
    #Sima
    target = co2_df["co2"]
    features = co2_df.drop(columns="co2")

    x_train, _, y_train, _ = train_test_split(
        features, target, test_size=0.2, random_state=32
    )

    if model_type == 'RandomForest':
        model = RandomForestRegressor(**RFR_PARAMS, random_state=32)
    elif model_type == 'LightGBM':
        model = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=32)
    elif model_type == 'XGBoost':
        model = xgb.XGBRegressor(**XGB_PARAMS, random_state=32, n_jobs=-1)
    else:
        raise ValueError("Invalid model type. Choose from 'RandomForest', 'LightGBM', or 'XGBoost'.")

    model.fit(x_train, y_train)
    return model
