RFR_SEARCH_SPACE = {
    "n_estimators": (50, 200),
    "max_depth": (5, 20),
    "min_samples_split": (2, 10),
    "min_samples_leaf": (1, 10),
    "max_leaf_nodes": (10, 100),
}

LGBM_SEARCH_SPACE = {
    "n_estimators": (100, 500),
    "learning_rate": (0.01, 0.2),
    "max_depth": (3, 10),
    "min_child_samples": (5, 30),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_alpha": (0.0, 1.0),
    "reg_lambda": (0.0, 1.0),
}

XGB_SEARCH_SPACE = {
    "n_estimators": (100, 500),
    "learning_rate": (0.01, 0.2),
    "max_depth": (3, 10),
    "min_child_weight": (1, 10),
    "gamma": (0.0, 5.0),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_alpha": (0.0, 1.0),
    "reg_lambda": (0.0, 1.0),
}

OPTUNA_TRIALS = 50
