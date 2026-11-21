import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from keras.callbacks import EarlyStopping
from lime.lime_tabular import RecurrentTabularExplainer
from scipy.linalg import lstsq
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from tensorflow import keras
from tensorflow.keras import layers

from data import db_handler as db
from models import trained_models as tm

TIME_SERIES_FEATURES = [
    "energy_per_capita",
    "energy_per_gdp",
    "primary_energy_consumption",
    "population",
    "gdp",
]


def predict_and_visualize_regressor(features: pd.DataFrame, target: pd.Series):
    _, x_test, _, y_test = train_test_split(features, target, test_size=0.2, random_state=32)

    random_forest_model = tm.get_model('RandomForest', features, target)
    rfr_predictions = random_forest_model.predict(x_test)
    print_feature_importance("Random Forest", random_forest_model, features)

    lightgbm_model = tm.get_model('LightGBM', features, target)
    lgbm_predictions = lightgbm_model.predict(x_test)
    print_feature_importance("LightGBM", lightgbm_model, features)

    xgboost_model = tm.get_model('XGBoost', features, target)
    xgb_predictions = xgboost_model.predict(x_test)
    print_feature_importance("XGBoost", xgboost_model, features)

    rfr_mse = mean_squared_error(y_test, rfr_predictions)
    rfr_mae = mean_absolute_error(y_test, rfr_predictions)
    print(f"Random Forest MSE: {rfr_mse}, MAE: {rfr_mae}")
    plot_results(y_test, rfr_predictions, "Random Forest")
    
    lgbm_mse = mean_squared_error(y_test, lgbm_predictions)
    lgbm_mae = mean_absolute_error(y_test, lgbm_predictions)
    print(f"LightGBM MSE: {lgbm_mse}, MAE: {lgbm_mae}")
    plot_results(y_test, lgbm_predictions, "LightGBM")

    xgb_mse = mean_squared_error(y_test, xgb_predictions)
    xgb_mae = mean_absolute_error(y_test, xgb_predictions)
    print(f"XGBoost MSE: {xgb_mse}, MAE: {xgb_mae}")
    plot_results(y_test, xgb_predictions, "XGBoost")


def print_feature_importance(model_name, model, features):
    importance = pd.Series(
        model.feature_importances_,
        index=features.columns,
    ).sort_values(ascending=False)

    print(f"\n{model_name} feature importance:")
    for feature_name, value in importance.items():
        print(f"  {feature_name}: {value:.6f}")


def plot_results(y_test, predictions, model_name):
    plt.figure(figsize=(15, 5))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--') 
    plt.xlabel('Actual CO2 Emissions')
    plt.ylabel('Predicted CO2 Emissions')
    plt.title(f'{model_name} Predictions')
    show_or_save(f"predictions_{model_name}")

def predict_and_visualize_time_series():
    df = db.get_from_db("gold_co2_time_series")
    x, y, target_years, target_countries = process_Xy(df)
    train_x = x[target_years <= 2005]
    train_y = y[target_years <= 2005]
    valid_x = x[(target_years > 2005) & (target_years <= 2014)]
    valid_y = y[(target_years > 2005) & (target_years <= 2014)]
    test_x = x[target_years > 2014]
    test_y = y[target_years > 2014]

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    n_features = train_x.shape[2]

    train_x_2d = train_x.reshape(-1, n_features)
    valid_x_2d = valid_x.reshape(-1, n_features)
    test_x_2d = test_x.reshape(-1, n_features)

    scaled_train = scaler_x.fit_transform(train_x_2d).reshape(train_x.shape)
    scaled_valid = scaler_x.transform(valid_x_2d).reshape(valid_x.shape)
    scaled_test = scaler_x.transform(test_x_2d).reshape(test_x.shape)

    target_train = scaler_y.fit_transform(train_y.reshape(-1, 1))
    target_valid = scaler_y.transform(valid_y.reshape(-1, 1))
    target_test = scaler_y.transform(test_y.reshape(-1, 1))

    model = keras.Sequential(
        [
            layers.Input(shape=train_x.shape[1:]),
            layers.GRU(32, activation="relu", input_shape = train_x.shape[1:]),
            layers.Dense(16),
            layers.Dense(8),
            layers.Dense(4),
            layers.Dense(1)
        ]
    )

    model.compile(
    loss="MeanSquaredError",
    optimizer=keras.optimizers.Adam(learning_rate=0.001)
)
    callbacks = [EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]
    history = model.fit(
            scaled_train,
            target_train,
            validation_data=(scaled_valid, target_valid),
            batch_size=16,
            epochs=100,
            callbacks=callbacks,
            shuffle=True,
            verbose=True,
        )


    test_mask = target_years > 2014

    test_years = target_years[test_mask]
    test_countries = target_countries[test_mask]

    country_mask = test_countries == "United States"

    pred_scaled = model.predict(scaled_test)
    pred = scaler_y.inverse_transform(pred_scaled).flatten()
    plt.figure(figsize=(10, 6))
    plt.plot(test_years[country_mask], test_y[country_mask], label="Real")
    plt.plot(test_years[country_mask], pred[country_mask], label="Predicted")
    plt.xticks(rotation=45)
    plt.title('Actual vs Predicted CO2 Emissions')
    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions')
    plt.legend()
    show_or_save("gru_time_series")

    year_to_explain = 2018
    country_to_explain = "Hungary"
    matching_rows = np.where(
        (test_years == year_to_explain) & (test_countries == country_to_explain)
    )[0]
    if len(matching_rows) == 0:
        print(
            f"Skipping LIME/SHAP explanation: no data for "
            f"{country_to_explain} in {year_to_explain}."
        )
        return
    row = matching_rows[0]

    explain_with_lime(model, scaled_train, target_train, scaled_test[row], scaler_y)
    explain_with_shap(model, scaled_train, scaled_test[row], scaler_y)


def explain_with_lime(model, training_data, training_labels, instance, target_scaler):
    explainer = RecurrentTabularExplainer(
        training_data=training_data,
        feature_names=TIME_SERIES_FEATURES,
        mode="regression",
        training_labels=training_labels.flatten(),
        random_state=42,
        discretize_continuous=True,
    )
    explanation = explainer.explain_instance(
        instance,
        lambda samples: target_scaler.inverse_transform(
            model.predict(samples, verbose=0)
        ).flatten(),
        num_features=30,
        num_samples=1000,
    )

    explanation.as_pyplot_figure()
    plt.tight_layout()
    show_or_save("lime_explanation")


def explain_with_shap(model, training_data, instance, target_scaler):
    background = training_data[
        np.random.default_rng(42).choice(
            len(training_data),
            size=min(100, len(training_data)),
            replace=False,
        )
    ]
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(instance[np.newaxis, ...])
    shap_values = np.array(shap_values).squeeze()

    n_timesteps = instance.shape[0]
    feature_names = [
        f"{feature}_t-{n_timesteps - 1 - timestep}"
        for timestep in range(n_timesteps)
        for feature in TIME_SERIES_FEATURES
    ]
    shap_values = shap_values.flatten() * target_scaler.scale_[0]

    base_value = (
        model.predict(background, verbose=0).mean()
        * target_scaler.scale_[0]
        + target_scaler.mean_[0]
    )

    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=instance.flatten(),
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, max_display=20)
    show_or_save("shap_explanation")

def process_Xy(df, lookback=10):
    df = df.copy()
    df = df.sort_values(["name", "year"]).reset_index(drop=True)

    features = [
        "gdp",
        "population",
        "energy_per_capita",
        "energy_per_gdp",
        "primary_energy_consumption"
    ]

    x, y, target_years, target_countries = [], [], [], []

    for country, group in df.groupby("name"):
        group = group.sort_values("year")

        if len(group) <= lookback:
            continue

        x_values = group[features].values.astype(np.float32)
        y_values = group["co2"].values.astype(np.float32)
        years = group["year"].values

        for i in range(lookback, len(group)):
            x.append(x_values[i - lookback:i])
            y.append(y_values[i])
            target_years.append(years[i])
            target_countries.append(country)

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    target_years = np.array(target_years)
    target_countries = np.array(target_countries)

    return x, y, target_years, target_countries


def print_surrogate_tree(features, target):
    model = tm.get_model('XGBoost', features, target)

    x_train, x_test, _, _ = train_test_split(features, target, test_size=0.2, random_state=32)

    y_train_for_surrogate = model.predict(x_train)
    y_test_for_surrogate = model.predict(x_test)

    surrogate_model = DecisionTreeRegressor(max_depth=5, random_state=32)
    surrogate_model.fit(x_train, y_train_for_surrogate)
    y_test_surrogate_pred = surrogate_model.predict(x_test)
    fidelity = r2_score(y_test_for_surrogate, y_test_surrogate_pred)

    print(f"Fidelity of surrogate model: {fidelity:.4f}")

    plt.figure(figsize=(20, 10))

    plot_tree(
        surrogate_model,
        feature_names=features.columns,
        filled=True,
        rounded=True,
        fontsize=8
    )

    plt.title("Surrogate Decision Tree")
    show_or_save("surrogate_tree")


def linear_regression(features, target):
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=32)

    # x_train = np.column_stack([np.ones(len(x_train)), x_train])
    # x_test = np.column_stack([np.ones(len(x_test)), x_test])

    coefficients, _, _, _ = lstsq(x_train, y_train)
    predictions = x_test @ coefficients
    
    print(f"Linear Regression R2: {r2_score(y_test, predictions)}")
    print(f"Linear Regression MSE: {mean_squared_error(y_test, predictions)}")
    print(f"Linear Regression MAE: {mean_absolute_error(y_test, predictions)}")
    print_metrics_by_emission_range(y_test, predictions)
    plot_results(y_test, predictions, "Linear Regression")
    show_or_save("linear_regression_results")


def print_metrics_by_emission_range(actual, predictions):
    emission_ranges = pd.cut(
        np.asarray(actual),
        bins=[0, 10, 100, 1000, 10000, float("inf")],
        labels=["0-10", "10-100", "100-1000", "1000-10000", "10000+"],
        right=False,
    )
    grouped_results = pd.DataFrame(
        {
            "actual": np.asarray(actual),
            "prediction": np.asarray(predictions),
            "emission_range": emission_ranges,
        }
    )

    print("Metrics by CO2 emission range:")
    for emission_range, group in grouped_results.groupby(
        "emission_range", observed=False
    ):
        if group.empty:
            continue

        mse = mean_squared_error(group["actual"], group["prediction"])
        mae = mean_absolute_error(group["actual"], group["prediction"])
        actual_mean = group["actual"].mean()
        actual_median = group["actual"].median()
        print(
            f"  {emission_range}: count={len(group)}, "
            f"MSE={mse}, MAE={mae}, "
            f"actual mean={actual_mean}, actual median={actual_median}"
        )


def plot_austria_predictions(features: pd.DataFrame, target: pd.Series):
    gold_co2_df = db.get_from_db("gold_co2")

    austria_mask = (
        gold_co2_df["name_Austria"].eq(1)
        & gold_co2_df["year"].between(1965, 1999)
    )
    austria_df = gold_co2_df.loc[austria_mask].sort_values("gdp_per_capita")

    austria_features = austria_df.drop(columns="co2")
    actual = austria_df["co2"]
    gdp_per_capita = np.expm1(austria_df["gdp_per_capita"])

    plt.figure(figsize=(15, 7))
    plt.plot(gdp_per_capita, actual, color="black", linewidth=2.5, marker="o",
             label="Actual")

    models = [
        ("Random Forest", tm.get_model("RandomForest", features, target)),
        ("LightGBM", tm.get_model("LightGBM", features, target)),
        ("XGBoost", tm.get_model("XGBoost", features, target)),
    ]

    for model_name, model in models:
        predictions = model.predict(austria_features)
        plt.plot(
            gdp_per_capita,
            predictions,
            linewidth=1.8,
            marker="o",
            markersize=3,
            label=model_name,
        )

    plt.xlabel("GDP per capita")
    plt.ylabel("CO2 emission")
    plt.title("Ausztria CO2 kibocsátása")
    plt.xticks(gdp_per_capita, rotation=45)
    plt.legend()
    plt.tight_layout()
    show_or_save("austria_1965_1999_predictions")
    sys.exit(0)

def predict_and_visualize(features: pd.DataFrame, target: pd.Series):
    plot_austria_predictions(features, target)
    predict_and_visualize_regressor(features, target)
    linear_regression(features, target)
    predict_and_visualize_time_series()
    print_surrogate_tree(features, target)


def show_or_save(filename: str) -> None:
    output_dir = os.getenv("OUTPUT_DIR")
    if not output_dir:
        plt.show()
        return

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    png = destination / f"{filename}.png"
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {png}")
