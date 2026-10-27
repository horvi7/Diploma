from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from models import trained_models as tm


def train_model(model, features, target):
    model.fit(features, target)


def print_metric_results(model_name, actual, predictions):
    mse = mean_squared_error(actual, predictions)
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)
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


def benchmark_model(model,features, target, sample_size, n_repeats):
    x_train, x_test, y_train, _ = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=32,
    )
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

    models = [
        ("Random Forest", tm.get_model("RandomForest", features, target)),
        ("LightGBM", tm.get_model("LightGBM", features, target)),
        ("XGBoost", tm.get_model("XGBoost", features, target)),
    ]

    for model_name, model in models:
        print_metric_results(model_name, y_test, model.predict(x_test))

    # Benchmark training time and prediction throughput.
    sample_size = 100
    n_repeats = 20

    for _, model in models:
        benchmark_model(model, features, target, sample_size, n_repeats)
