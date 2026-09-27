import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split

import db_handler as db

rfr_params = {'n_estimators': 105, 'max_depth': 6, 'min_samples_split': 9, 'min_samples_leaf': 4, 'bootstrap': True, 'max_leaf_nodes': 24}
lgbm_params = {'n_estimators': 167, 'learning_rate': 0.011882888130545924, 'max_depth': 4, 'min_child_samples': 23, 'subsample': 0.9390586696763323, 'colsample_bytree': 0.6506930125803669, 'reg_alpha': 0.9821512272350051, 'reg_lambda': 0.11827601292607956}
xgb_params = {'n_estimators': 381, 'learning_rate': 0.11854834150995305, 'max_depth': 6, 'min_child_weight': 5, 'gamma': 4.4277220131081, 'subsample': 0.9850997649876527, 'colsample_bytree': 0.6318153394672948, 'reg_alpha': 0.4635012413919827, 'reg_lambda': 0.10254642573937399}

def get_model(model_type):
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

    x_train, _, y_train, _ = train_test_split(
        features, target, test_size=0.2, random_state=32
    )

    if model_type == 'RandomForest':
        model = RandomForestRegressor(**rfr_params, random_state=32)
    elif model_type == 'LightGBM':
        model = lgb.LGBMRegressor(**lgbm_params, random_state=32)
    elif model_type == 'XGBoost':
        model = xgb.XGBRegressor(**xgb_params, random_state=32, n_jobs=-1)
    else:
        raise ValueError("Invalid model type. Choose from 'RandomForest', 'LightGBM', or 'XGBoost'.")

    model.fit(x_train, y_train)
    return model