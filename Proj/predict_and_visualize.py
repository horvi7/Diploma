import train_models_with_optuna as train_optuna
import db_handler as db
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def predict_and_visualize():
    co2_df = db.get_from_db("gold_co2")

    #make test and train split
    target = co2_df.iloc[:, 0]
    features = co2_df.iloc[:, 1:]

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=32)
    #A pipelinehoz ez kell majd de hogy ne menjen végig mindig, maunálisan a legjobb paramokkal futtatom.
    #rfr_params, lgbm_params, xgb_params = train_optuna.get_best_params_for_models(features, target)

    rfr_params = {'n_estimators': 105, 'max_depth': 6, 'min_samples_split': 9, 'min_samples_leaf': 4, 'bootstrap': True, 'max_leaf_nodes': 24}
    lgbm_params = {'n_estimators': 167, 'learning_rate': 0.011882888130545924, 'max_depth': 4, 'min_child_samples': 23, 'subsample': 0.9390586696763323, 'colsample_bytree': 0.6506930125803669, 'reg_alpha': 0.9821512272350051, 'reg_lambda': 0.11827601292607956}
    xgb_params = {'n_estimators': 381, 'learning_rate': 0.11854834150995305, 'max_depth': 6, 'min_child_weight': 5, 'gamma': 4.4277220131081, 'subsample': 0.9850997649876527, 'colsample_bytree': 0.6318153394672948, 'reg_alpha': 0.4635012413919827, 'reg_lambda': 0.10254642573937399}

    random_forest_model = RandomForestRegressor(**rfr_params, random_state=32)
    random_forest_model.fit(x_train, y_train)
    rfr_predictions = random_forest_model.predict(x_test)
    print("Random Forest Predictions:", rfr_predictions)

    lightgbm_model = lgb.LGBMRegressor(**lgbm_params, random_state=32)
    lightgbm_model.fit(x_train, y_train)
    lgbm_predictions = lightgbm_model.predict(x_test)
    print("LightGBM Predictions:", lgbm_predictions)

    xgboost_model = xgb.XGBRegressor(**xgb_params, random_state=32, n_jobs=-1)
    xgboost_model.fit(x_train, y_train)
    xgb_predictions = xgboost_model.predict(x_test)
    print("XGBoost Predictions:", xgb_predictions)

    plt.figure(figsize=(15, 5))
    plt.scatter(y_test, rfr_predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--') 
    plt.xlabel('Actual CO2 Emissions')
    plt.ylabel('Predicted CO2 Emissions')
    plt.title('Random Forest Predictions')
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.scatter(y_test, lgbm_predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--') 
    plt.xlabel('Actual CO2 Emissions')
    plt.ylabel('Predicted CO2 Emissions') 
    plt.title('LightGBM Predictions')
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.scatter(y_test, xgb_predictions, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--')
    plt.xlabel("Actual CO2")
    plt.ylabel("Predicted CO2")
    plt.title("Actual vs Predicted")
    plt.show()

predict_and_visualize()