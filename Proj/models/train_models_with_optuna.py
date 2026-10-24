import lightgbm as lgb
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split

from models.model_parameters import LGBM_PARAMS, RFR_PARAMS, XGB_PARAMS
from models.optuna_parameters import (
    LGBM_SEARCH_SPACE,
    OPTUNA_TRIALS,
    RFR_SEARCH_SPACE,
    XGB_SEARCH_SPACE,
)

MANUAL_PARAMS = {
    "RandomForest": {
    "n_estimators": 50,
    "max_depth": 6,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "bootstrap": True,
    "max_leaf_nodes": 30
    },

    "LightGBM": {
    "n_estimators": 70,
    "learning_rate": 0.01,
    "max_depth": 4,
    "min_child_samples": 12,
    "subsample": 0.70,
    "colsample_bytree": 0.72,
    "reg_alpha": 0.74,
    "reg_lambda": 0.21,
    },

    "XGBoost": { "n_estimators": 20,
    "learning_rate": 0.2,
    "max_depth": 4,
    "min_child_weight": 4,
    "gamma": 3.2,
    "subsample": 0.93,
    "colsample_bytree": 0.69,
    "reg_alpha": 0.38,
    "reg_lambda": 0.3
    },
}


def objective_RFR(trial, features, target):
    regressor = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", *RFR_SEARCH_SPACE["n_estimators"]),
        max_depth=trial.suggest_int("max_depth", *RFR_SEARCH_SPACE["max_depth"]),
        min_samples_split=trial.suggest_int(
            "min_samples_split", *RFR_SEARCH_SPACE["min_samples_split"]
        ),
        min_samples_leaf=trial.suggest_int(
            "min_samples_leaf", *RFR_SEARCH_SPACE["min_samples_leaf"]
        ),
        bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        max_leaf_nodes=trial.suggest_int(
            "max_leaf_nodes", *RFR_SEARCH_SPACE["max_leaf_nodes"]
        ),
        random_state=32,
    )
    scores = cross_val_score(regressor, features, target, cv=3, scoring="r2")
    return scores.mean()


def objective_LGBM(trial, features, target):
    model = lgb.LGBMRegressor(
        n_estimators=trial.suggest_int(
            "n_estimators", *LGBM_SEARCH_SPACE["n_estimators"]
        ),
        learning_rate=trial.suggest_float(
            "learning_rate", *LGBM_SEARCH_SPACE["learning_rate"]
        ),
        max_depth=trial.suggest_int("max_depth", *LGBM_SEARCH_SPACE["max_depth"]),
        min_child_samples=trial.suggest_int(
            "min_child_samples", *LGBM_SEARCH_SPACE["min_child_samples"]
        ),
        subsample=trial.suggest_float("subsample", *LGBM_SEARCH_SPACE["subsample"]),
        colsample_bytree=trial.suggest_float(
            "colsample_bytree", *LGBM_SEARCH_SPACE["colsample_bytree"]
        ),
        reg_alpha=trial.suggest_float("reg_alpha", *LGBM_SEARCH_SPACE["reg_alpha"]),
        reg_lambda=trial.suggest_float(
            "reg_lambda", *LGBM_SEARCH_SPACE["reg_lambda"]
        ),
        random_state=32,
        n_jobs=-1,
    )
    scores = cross_val_score(model, features, target, cv=3, scoring="r2")
    return scores.mean()


def objective_XGB(trial, features, target):
    model = xgb.XGBRegressor(
        n_estimators=trial.suggest_int(
            "n_estimators", *XGB_SEARCH_SPACE["n_estimators"]
        ),
        learning_rate=trial.suggest_float(
            "learning_rate", *XGB_SEARCH_SPACE["learning_rate"]
        ),
        max_depth=trial.suggest_int("max_depth", *XGB_SEARCH_SPACE["max_depth"]),
        min_child_weight=trial.suggest_int(
            "min_child_weight", *XGB_SEARCH_SPACE["min_child_weight"]
        ),
        gamma=trial.suggest_float("gamma", *XGB_SEARCH_SPACE["gamma"]),
        subsample=trial.suggest_float("subsample", *XGB_SEARCH_SPACE["subsample"]),
        colsample_bytree=trial.suggest_float(
            "colsample_bytree", *XGB_SEARCH_SPACE["colsample_bytree"]
        ),
        reg_alpha=trial.suggest_float("reg_alpha", *XGB_SEARCH_SPACE["reg_alpha"]),
        reg_lambda=trial.suggest_float(
            "reg_lambda", *XGB_SEARCH_SPACE["reg_lambda"]
        ),
        random_state=32,
        n_jobs=-1,
        verbosity=0,
    )
    scores = cross_val_score(model, features, target, cv=3, scoring="r2")
    return scores.mean()


def get_best_params_for_models(features, target):
    studies = {
        "RandomForest": (objective_RFR, "Random Forest"),
        "LightGBM": (objective_LGBM, "LightGBM"),
        "XGBoost": (objective_XGB, "XGBoost"),
    }
    best_params = {}

    for model_name, (objective, display_name) in studies.items():
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, features, target),
            n_trials=OPTUNA_TRIALS,
        )
        best_params[model_name] = study.best_params
        print(f"Optuna {display_name} best CV R2: {study.best_value:.4f}")

    return best_params


def compare_manual_and_optuna(features: pd.DataFrame, target: pd.Series):
    """Compare manual and saved Optuna parameters by training R2."""
    x_train, _, y_train, _ = train_test_split(
        features, target, test_size=0.2, random_state=32
    )
    parameter_sets = {
        "Manual": MANUAL_PARAMS,
        "Optuna": {
            "RandomForest": RFR_PARAMS,
            "LightGBM": LGBM_PARAMS,
            "XGBoost": XGB_PARAMS,
        },
    }

    print("\nManual vs Optuna training R2:")
    results = {}
    for model_name in ("RandomForest", "LightGBM", "XGBoost"):
        results[model_name] = {}
        for parameter_name, parameters in parameter_sets.items():
            model = _create_model(model_name, parameters[model_name])
            model.fit(x_train, y_train)
            score = model.score(x_train, y_train)
            results[model_name][parameter_name] = score
            print(f"{model_name} ({parameter_name}): R2={score:.4f}")

    return results


def _create_model(model_name, parameters):
    if model_name == "RandomForest":
        return RandomForestRegressor(**parameters, random_state=32)
    if model_name == "LightGBM":
        return lgb.LGBMRegressor(**parameters, random_state=32, n_jobs=-1)
    if model_name == "XGBoost":
        return xgb.XGBRegressor(
            **parameters,
            random_state=32,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model name: {model_name}")
