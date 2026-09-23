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

  custom_feature = pd.DataFrame([{
      "log_population": 16.656126,
      "log_gdp_per_population": 10.207357,
      "log_energy_per_gdp": 2.265,
      "log_co2_per_energy": 0.004512,
  }])

  prediction = model.predict(custom_feature)
  print(f"Prediction for custom feature: {prediction}")

  decrease_range_low = 0.8
  decrease_range_high = 0.9

  cf = exp.generate_counterfactuals(custom_feature, total_CFs=1, desired_range=[prediction[0]*decrease_range_low , prediction[0]*decrease_range_high], features_to_vary=features_kaya.columns.tolist())

  # A generált forgatókönyv kinyerése DataFrame-ként
  cf_df = cf.cf_examples_list[0].final_cfs_df

  print("\n--- Generált ellenirányú forgatókönyv (Counterfactual) ---")
  print(cf_df)

analyze_decision()