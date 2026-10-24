import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from data import db_handler as db
from models.model_parameters import LGBM_PARAMS, RFR_PARAMS, XGB_PARAMS


def get_model(model_type: str, features: pd.DataFrame, target: pd.Series):
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
