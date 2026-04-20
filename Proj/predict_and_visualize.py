import train_models_with_optuna as train_optuna
import db_handler as db

def predict_and_visualize():
    co2_df = db.get_from_db("gold_co2_emissions")
    target = co2_df.iloc[:, 0]
    features = co2_df.iloc[:, 1:]

    rfr_params, lgbm_params, xgb_params = train_optuna.get_best_params_for_models(features, target)

