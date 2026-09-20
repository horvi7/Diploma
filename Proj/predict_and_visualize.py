from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras, reshape
import train_models_with_optuna as train_optuna
import db_handler as db
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, activations
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd

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


    rfr_mse = mean_squared_error(y_test, rfr_predictions)
    rfr_mae = mean_absolute_error(y_test, rfr_predictions)
    print(f"Random Forest MSE: {rfr_mse}, MAE: {rfr_mae}")

    plt.figure(figsize=(15, 5))
    plt.scatter(y_test, rfr_predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--') 
    plt.xlabel('Actual CO2 Emissions')
    plt.ylabel('Predicted CO2 Emissions')
    plt.title('Random Forest Predictions')
    plt.show()


    lgbm_mse = mean_squared_error(y_test, lgbm_predictions)
    lgbm_mae = mean_absolute_error(y_test, lgbm_predictions)
    print(f"LightGBM MSE: {lgbm_mse}, MAE: {lgbm_mae}")

    plt.figure(figsize=(15, 5))
    plt.scatter(y_test, lgbm_predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--') 
    plt.xlabel('Actual CO2 Emissions')
    plt.ylabel('Predicted CO2 Emissions') 
    plt.title('LightGBM Predictions')
    plt.show()

    xgb_mse = mean_squared_error(y_test, xgb_predictions)
    xgb_mae = mean_absolute_error(y_test, xgb_predictions)
    print(f"XGBoost MSE: {xgb_mse}, MAE: {xgb_mae}")

    plt.figure(figsize=(15, 5))
    plt.scatter(y_test, xgb_predictions, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--')
    plt.xlabel("Actual CO2")
    plt.ylabel("Predicted CO2")
    plt.title("Actual vs Predicted")
    plt.show()

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
    plt.show()

  

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


predict_and_visualize()
#predict_and_visualize_time_series()