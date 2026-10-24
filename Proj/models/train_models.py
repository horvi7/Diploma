import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from data import db_handler as db


#Ez nem nagyon van használva, mert ez csak az eredeti manuális próbálkozás, de aztán optuna be lett vezetve, majd a paraméterek elmentve
def train_random_forest(features, target):
  feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=32)

  regressor = RandomForestRegressor(
    n_estimators=105,
    max_depth=6,
    min_samples_split=9,
    min_samples_leaf=4,
    bootstrap=True,
    max_leaf_nodes=24,
    random_state=32
)
  regressor.fit(feature_train, target_train)
  
  target_predicted = regressor.predict(feature_test)
  result_df = pd.DataFrame({"Actual": target_test, "Predicted": target_predicted})
  print(result_df)
  print("Score:", regressor.score(feature_test, target_test))
  return regressor.score(feature_test, target_test)


def train_xgboost(features, target):
  feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=32)

  regressor = xgb.XGBRegressor(
    n_estimators= 381,
    learning_rate= 0.11854834150995305,
    max_depth= 6,
    min_child_weight= 5,
    gamma= 4.4277220131081,
    subsample= 0.9850997649876527,
    colsample_bytree= 0.6318153394672948,
    reg_alpha= 0.4635012413919827,
    reg_lambda= 0.10254642573937399,
    random_state=32,
    n_jobs=-1,
)
  regressor.fit(feature_train, target_train)
  
  target_predicted = regressor.predict(feature_test)
  result_df = pd.DataFrame({"Actual": target_test, "Predicted": target_predicted})
  print(result_df)
  print("Score:", regressor.score(feature_test, target_test))
  return regressor.score(feature_test, target_test)

def train_lightgbm(features, target):
  feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=32)

  regressor = lgb.LGBMRegressor(
    n_estimators=105,
    learning_rate= 0.011882888130545924,
    max_depth= 4,
    min_child_samples= 23,
    subsample= 0.9390586696763323,
    colsample_bytree= 0.6506930125803669,
    reg_alpha= 0.9821512272350051,
    reg_lambda= 0.11827601292607956,
    random_state=32,
    n_jobs=-1,
)
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
