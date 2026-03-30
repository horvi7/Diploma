from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import db_handler as db
import xgboost as xgb
import lightgbm as lgb
import optuna

# def objective(trial, features, target):
#   n_estimators = trial.suggest_int('n_estimators', 50, 200)
#   max_depth = trial.suggest_int('max_depth', 5, 20)
#   min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
#   min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
#   bootstrap = trial.suggest_categorical('bootstrap', [True, False])
#   max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 10, 100)
  
#   regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_leaf_nodes=max_leaf_nodes, random_state=32)
  
#   scores = cross_val_score(regressor, features,  target, cv=3,  scoring="r2") 

#   return scores.mean()

# def objective(trial, features, target):
#   model = lgb.LGBMRegressor(
#     n_estimators=trial.suggest_int('n_estimators', 100, 500),
#     learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2),
#     max_depth = trial.suggest_int('max_depth', 3, 10),
#     min_child_samples=trial.suggest_int('min_child_samples', 5, 30),
#     subsample=trial.suggest_float('subsample', 0.6, 1.0),
#     colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
#     reg_alpha=trial.suggest_float('reg_alpha', 0.0, 1.0),
#     reg_lambda=trial.suggest_float('reg_lambda', 0.0, 1.0),
#     random_state=32,
#     n_jobs=-1
#   )

#   scores = cross_val_score(model, features, target, cv=3, scoring="r2")

#   return scores.mean()


def objective(trial, features, target):
  model = xgb.XGBRegressor(
        n_estimators=trial.suggest_int('n_estimators', 100, 500),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
        gamma=trial.suggest_float('gamma', 0.0, 5.0),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
        reg_alpha=trial.suggest_float('reg_alpha', 0.0, 1.0),
        reg_lambda=trial.suggest_float('reg_lambda', 0.0, 1.0),
        random_state=32,
        n_jobs=-1,
        verbosity=0
        )

  scores = cross_val_score(
      model,
      features,
      target,
      cv=3,
      scoring="r2"
  )

  return scores.mean()

co2_df = db.get_from_db("gold_co2_emissions")
print("Itt: ", co2_df)
target = co2_df.iloc[:, 0]
features = co2_df.iloc[:, 1:]

# study = optuna.create_study(direction='maximize')
# study.optimize(lambda trial: objective(trial, features, target), n_trials=100)
# print(study.best_params)
best_params = {'n_estimators': 103, 'learning_rate': 0.010960626026507832, 'max_depth': 3, 'min_child_weight': 8, 'gamma': 4.028943921317211, 'subsample': 0.6965817199253909, 'colsample_bytree': 0.7589653328612164, 'reg_alpha': 0.866112422886934, 'reg_lambda': 0.2543198152148851}

model = xgb.XGBRegressor(**best_params, random_state=32, n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=32)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("R2:", r2)
print("RMSE:", rmse)
print("MAE:", mae)

# importances = model.feature_importances_
# feature_importance_df = pd.DataFrame({
#     "feature": features.columns,
#     "importance": importances
# }).sort_values(by="importance", ascending=False)
print("importance: ", model.get_booster().get_score(importance_type="gain"))

#train_random_forest(features, target)
# train_xgboost(features, target)
# train_lightgbm(features, target)
