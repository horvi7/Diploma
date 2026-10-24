from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from models import trained_models as tm


def train_model(model, features, target):
    model.fit(features, target)


def print_metric_results(model_name, mse, mae, r2):
    print(f"{model_name} MSE: {mse}, MAE: {mae}, R2: {r2}")


def print_results(model_name, train_times, throughput_values):
    print(f"\n{model_name}")
    print(
        f"Training time: "
        f"{np.mean(train_times):.4f} +- "
        f"{np.std(train_times):.4f} seconds | "
        f"Sum: {np.sum(train_times):.4f} seconds"
    )
    print(
        f"Throughput: "
        f"{np.mean(throughput_values):.2f} +- "
        f"{np.std(throughput_values):.2f} predictions/sec"
    )


def benchmark_model(model, x_train, y_train, x_test, sample_size, n_repeats):
    train_times = []
    throughput_values = []

    # Measure training time.
    for _ in range(n_repeats):
        model_copy = clone(model)
        start = perf_counter()
        model_copy.fit(x_train, y_train)

        train_time = perf_counter() - start
        train_times.append(train_time)

    # Measure prediction throughput.
    model_copy = clone(model)
    model_copy.fit(x_train, y_train)
    x_sample = x_test[:sample_size]

    # Warm up the model before measuring.
    model_copy.predict(x_sample)

    for _ in range(n_repeats):
        start = perf_counter()
        model_copy.predict(x_sample)
        prediction_time = perf_counter() - start

        throughput = len(x_sample) / prediction_time
        throughput_values.append(throughput)

    print_results(model.__class__.__name__, train_times, throughput_values)


def make_metrics(features: pd.DataFrame, target: pd.Series):
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=32,
    )

    # Evaluate model predictions.
    random_forest_model = tm.get_model("RandomForest", features, target)
    rfr_predictions = random_forest_model.predict(x_test)

    lightgbm_model = tm.get_model("LightGBM", features, target)
    lgbm_predictions = lightgbm_model.predict(x_test)

    xgboost_model = tm.get_model("XGBoost", features, target)
    xgb_predictions = xgboost_model.predict(x_test)

    rfr_mse = mean_squared_error(y_test, rfr_predictions)
    rfr_mae = mean_absolute_error(y_test, rfr_predictions)
    rfr_r2 = r2_score(y_test, rfr_predictions)

    lgbm_mse = mean_squared_error(y_test, lgbm_predictions)
    lgbm_mae = mean_absolute_error(y_test, lgbm_predictions)
    lgbm_r2 = r2_score(y_test, lgbm_predictions)

    xgb_mse = mean_squared_error(y_test, xgb_predictions)
    xgb_mae = mean_absolute_error(y_test, xgb_predictions)
    xgb_r2 = r2_score(y_test, xgb_predictions)

    print_metric_results("Random Forest", rfr_mse, rfr_mae, rfr_r2)
    print_metric_results("LightGBM", lgbm_mse, lgbm_mae, lgbm_r2)
    print_metric_results("XGBoost", xgb_mse, xgb_mae, xgb_r2)

    # Benchmark training time and prediction throughput.
    models = [random_forest_model, lightgbm_model, xgboost_model]
    sample_size = 100
    n_repeats = 20

    for model in models:
        benchmark_model(model, x_train, y_train, x_test, sample_size, n_repeats)
