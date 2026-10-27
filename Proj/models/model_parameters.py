MANUAL_PARAMS = {
    "RandomForest": {
        "n_estimators": 105,
        "max_depth": 6,
        "min_samples_split": 9,
        "min_samples_leaf": 4,
        "bootstrap": True,
        "max_leaf_nodes": 24
    },
    "LightGBM": {
        "n_estimators": 105,
        "learning_rate": 0.011882888130545924,
        "max_depth": 4,
        "min_child_samples": 23,
        "subsample": 0.9390586696763323,
        "colsample_bytree": 0.6506930125803669,
        "reg_alpha": 0.9821512272350051,
        "reg_lambda": 0.11827601292607956
    },
    "XGBoost": {
        "n_estimators": 381,
        "learning_rate": 0.11854834150995305,
        "max_depth": 6,
        "min_child_weight": 5,
        "gamma": 4.4277220131081,
        "subsample": 0.9850997649876527,
        "colsample_bytree": 0.6318153394672948,
        "reg_alpha": 0.4635012413919827,
        "reg_lambda": 0.10254642573937399
    },
}

RFR_PARAMS = {'n_estimators': 81, 'max_depth': 13, 'min_samples_split': 10, 'min_samples_leaf': 1, 'bootstrap': True, 'max_leaf_nodes': 63}

LGBM_PARAMS = {'n_estimators': 146, 'learning_rate': 0.1894163467997181, 'max_depth': 10, 'min_child_samples': 7, 'subsample': 0.6012940387570913, 'colsample_bytree': 0.7307865541286798, 'reg_alpha': 0.8960024018173693, 'reg_lambda': 0.19625297013684237}

XGB_PARAMS = {'n_estimators': 229, 'learning_rate': 0.19844298446228187, 'max_depth': 7, 'min_child_weight': 6, 'gamma': 2.653413357911681, 'subsample': 0.6061972977699855, 'colsample_bytree': 0.8364115718996126, 'reg_alpha': 0.8616111211196239, 'reg_lambda': 0.35457994999219394}
