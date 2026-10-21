import trained_models as tm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from time import perf_counter
from sklearn.base import clone
import numpy as np

def train_model(model, features, target):
    model.fit(features, target)

def make_metrics(features, target):
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=32)

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

    # Time for predictions

    models = [random_forest_model, lightgbm_model, xgboost_model]

    sample_size = 100
    n_repeats = 20

    for model in models:

        train_times = []
        throughput_values = []
        memory_usages = [] # to be done

        #Training time
        for _ in range(n_repeats):
            model_copy = clone(model)
            start = perf_counter()
            model_copy.fit(x_train, y_train)

            train_time = perf_counter() - start
            train_times.append(train_time)

        #Throughput
        model_copy = clone(model)
        model_copy.fit(x_train, y_train)
        x_sample = x_test[:sample_size]

        # warm-up
        model_copy.predict(x_sample)

        for _ in range(n_repeats):
            start = perf_counter()
            model_copy.predict(x_sample)
            prediction_time = perf_counter() - start

            throughput = len(x_sample) / prediction_time
            throughput_values.append(throughput)

        # Printing results
        print(f"\n{model.__class__.__name__}")
        print(f"Training time: "
            f"{np.mean(train_times):.4f} +- "
            f"{np.std(train_times):.4f} seconds | "
            f"Sum: {np.sum(train_times):.4f} seconds"
        )

        print(f"Throughput: "
            f"{np.mean(throughput_values):.2f} +- "
            f"{np.std(throughput_values):.2f} predictions/sec"
        )


