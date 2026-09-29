import trained_models as tm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def make_metrics(features, target):
    _, x_test, _, y_test = train_test_split(features, target, test_size=0.2, random_state=32)

    random_forest_model = tm.get_model('RandomForest')
    rfr_predictions = random_forest_model.predict(x_test)

    lightgbm_model = tm.get_model('LightGBM')
    lgbm_predictions = lightgbm_model.predict(x_test)

    xgboost_model = tm.get_model('XGBoost')
    xgb_predictions = xgboost_model.predict(x_test)

    rfr_mse = mean_squared_error(y_test, rfr_predictions)
    rfr_mae = mean_absolute_error(y_test, rfr_predictions)
    rfr_r2 = r2_score(y_test, rfr_predictions)

    lgbm_mse = mean_squared_error(y_test, lgbm_predictions)
    lgbm_mae = mean_absolute_error(y_test, lgbm_predictions)
    lgbm_r2 = r2_score(y_test, lgbm_predictions)

    xgb_mse = mean_squared_error(y_test, xgb_predictions)
    xgb_mae =  mean_absolute_error(y_test, xgb_predictions)
    xgb_r2 = r2_score(y_test, xgb_predictions)

    print(f"Random Forest MSE: {rfr_mse}, MAE: {rfr_mae}, R2: {rfr_r2}")
    print(f"LightGBM MSE: {lgbm_mse}, MAE: {lgbm_mae}, R2: {lgbm_r2}")
    print(f"XGBoost: {xgb_mse}, MAE: {xgb_mae}, R2: {xgb_r2}")
