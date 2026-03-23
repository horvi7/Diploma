from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import db_handler as db
import xgboost as xgb
import lightgbm as lgb
import optuna

# def objective(trial, features, target):
#   feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=32)

#   n_estimators = trial.suggest_int('n_estimators', 50, 200)
#   max_depth = trial.suggest_int('max_depth', 5, 20)
  
#   regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=32)
#   regressor.fit(feature_train, target_train)
  
#   score = regressor.score(feature_test, target_test)
#   return score


def objective(trial, features, target):
  feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=32)

  n_estimators = trial.suggest_int('n_estimators', 50, 200)
  learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
  
  regressor = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=32)
  regressor.fit(feature_train, target_train)
  
  # score = regressor.score(feature_test, target_test)
  # score = regressor.cros
  return score

def train_xgboost(features, target):
  feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=32)

  regressor = xgb.XGBRegressor(n_estimators=100, random_state=32)
  regressor.fit(feature_train, target_train)
  
  target_predicted = regressor.predict(feature_test)
  result_df = pd.DataFrame({"Actual": target_test, "Predicted": target_predicted})
  print(result_df)
  print("Score:", regressor.score(feature_test, target_test))




co2_df = db.get_from_db("gold_co2_emissions")
target = co2_df.iloc[:, 0]
features = co2_df.iloc[:, 1:]

study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, features, target), n_trials=50)
print(study.best_params)

#train_random_forest(features, target)
# train_xgboost(features, target)
# train_lightgbm(features, target)
