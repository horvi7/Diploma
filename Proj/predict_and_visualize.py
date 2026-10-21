import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
import shap
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from lime.lime_tabular import RecurrentTabularExplainer
import db_handler as db
import trained_models as tm

def predict_and_visualize_regressor(features, target):
    _, x_test, _, y_test = train_test_split(features, target, test_size=0.2, random_state=32)

    random_forest_model = tm.get_model('RandomForest')
    rfr_predictions = random_forest_model.predict(x_test)

    lightgbm_model = tm.get_model('LightGBM')
    lgbm_predictions = lightgbm_model.predict(x_test)

    xgboost_model = tm.get_model('XGBoost')
    xgb_predictions = xgboost_model.predict(x_test)

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

def plot_results(y_test, predictions, model_name):
    plt.figure(figsize=(15, 5))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--') 
    plt.xlabel('Actual CO2 Emissions')
    plt.ylabel('Predicted CO2 Emissions')
    plt.title(f'{model_name} Predictions')
    plt.show()

def predict_and_visualize_time_series():
    df = db.get_from_db("gold_co2_time_series")
    print("Df: ", df.head())
    x, y, target_years, target_countries = process_Xy(df)
    print("df cols: ", df.columns.tolist())
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
    plt.show()


    #LIME magyarázat
    features = ['energy_per_capita', 'energy_per_gdp', 'primary_energy_consumption', 'log_population', 'log_gdp']

    explainer = RecurrentTabularExplainer(
        training_data=scaled_train,
        feature_names=features,
        mode='regression',
        training_labels=target_train.flatten(),
        random_state=42,
        discretize_continuous=True #Enélkül is próba
    )

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
    explanation = explainer.explain_instance(
    scaled_test[row],
    lambda samples: scaler_y.inverse_transform(
        model.predict(samples, verbose=0)
    ).flatten(),
    num_features=30, #Mennyi magyarázatot ad ki lokálisan a minta körül.
    num_samples=1000)

    fig = explanation.as_pyplot_figure()
    plt.tight_layout()
    plt.show()

    # SHAP magyarázat

    # 1. Background adatok
    background = scaled_train[
        np.random.default_rng(42).choice(
            len(scaled_train),
            size=min(100, len(scaled_train)),
            replace=False
        )
    ]

    # 2. SHAP explainer
    explainer = shap.GradientExplainer(model, background) #Íráshoz: link hogy miért ez.

    # 3. Kiválasztott megfigyelés magyarázata
    shap_values = explainer.shap_values(scaled_test[row:row+1])

    shap_values = np.array(shap_values).squeeze()

    # 4. Feature-nevek időlépésekkel
    n_timesteps = scaled_test.shape[1]
    #Csak h mit írjon ki a magyarázatban
    feature_names = [
        f"{feature}_t-{n_timesteps - 1 - timestep}"
        for timestep in range(n_timesteps)
        for feature in features
    ]

    # 5. SHAP értékek visszaskálázása eredeti CO2-skálára
    shap_values = shap_values.flatten() * scaler_y.scale_[0]

    base_value = (
        model.predict(background, verbose=0).mean()
        * scaler_y.scale_[0]
        + scaler_y.mean_[0]
    )

    # 6. Ábra
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=scaled_test[row].flatten(),
        feature_names=feature_names
    )

    shap.plots.waterfall(explanation, max_display=20)

def process_Xy(df, lookback=10):
    df = df.copy()
    df = df.sort_values(["name", "year"]).reset_index(drop=True)

    features = [
        "log_gdp",
        "log_population",
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
    model = tm.get_model('XGBoost')

    x_train, x_test, _, _ = train_test_split(features, target, test_size=0.2, random_state=32)

    y_train_for_surrogate = model.predict(x_train)
    y_test_for_surrogate = model.predict(x_test)

    surrogate_model = DecisionTreeRegressor(max_depth=3, random_state=32)
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
    plt.show()

if __name__ == "__main__":
    print_surrogate_tree()