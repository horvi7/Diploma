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
import dice_ml

def analyze_decision():
  df = db.get_from_db("gold_co2")
  print(df.head())

  target = df["co2"]
  features_kaya = df[[
      "log_population",
      "log_gdp_per_population",
      "log_energy_per_gdp",
      "log_co2_per_energy",
  ]]
  df_kaya = df[[
      "co2",
      "log_population",
      "log_gdp_per_population",
      "log_energy_per_gdp",
      "log_co2_per_energy",
  ]]

  features_all = df.drop(columns=["co2"])

  dice_dataset_kaya = dice_ml.Data(dataframe=df_kaya, continuous_features=features_kaya.columns.tolist(), outcome_name="co2")

  dice_dataset_all = dice_ml.Data(dataframe=df, continuous_features=features_all.columns.tolist(), outcome_name="co2")

  model = RandomForestRegressor(n_estimators=105, max_depth=6, min_samples_split=9, min_samples_leaf=4, bootstrap=True, max_leaf_nodes=24, random_state=32)
  model.fit(features_kaya, target)

  dice_model = dice_ml.Model(model=model, backend="sklearn",model_type="regressor")

  exp = dice_ml.Dice(dice_dataset_kaya, dice_model)

analyze_decision()