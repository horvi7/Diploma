import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import db_handler as db

#Ez nem nagyon van használva, mert ez csak az eredeti manuális próbálkozás, de aztán optuna be lett vezetve, majd a paraméterek elmentve
def train_random_forest(features, target):
  feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=32)

  regressor = RandomForestRegressor(n_estimators=100, random_state=32)
  regressor.fit(feature_train, target_train)
  
  target_predicted = regressor.predict(feature_test)
  result_df = pd.DataFrame({"Actual": target_test, "Predicted": target_predicted})
  print(result_df)
  print("Score:", regressor.score(feature_test, target_test))
  return regressor.score(feature_test, target_test)


def train_xgboost(features, target):
  feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=32)

  regressor = xgb.XGBRegressor(n_estimators=100, random_state=32)
  regressor.fit(feature_train, target_train)
  
  target_predicted = regressor.predict(feature_test)
  result_df = pd.DataFrame({"Actual": target_test, "Predicted": target_predicted})
  print(result_df)
  print("Score:", regressor.score(feature_test, target_test))
  return regressor.score(feature_test, target_test)

def train_lightgbm(features, target):
  feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=32)

  regressor = lgb.LGBMRegressor(n_estimators=100, random_state=32, learning_rate=0.1)
  regressor.fit(feature_train, target_train, )
  
  target_predicted = regressor.predict(feature_test)
  result_df = pd.DataFrame({"Actual": target_test, "Predicted": target_predicted})
  print(result_df)
  print("Score:", regressor.score(feature_test, target_test))
  return regressor.score(feature_test, target_test)

def train_legacy_models():
  co2_df = db.get_from_db("gold_co2")
  if co2_df is None:
    return {}

  target = co2_df.iloc[:, 0]
  features = co2_df.iloc[:, 1:]

  return {
    "RandomForest": train_random_forest(features, target),
    "XGBoost": train_xgboost(features, target),
    "LightGBM": train_lightgbm(features, target),
  }


if __name__ == "__main__":
  train_legacy_models()
