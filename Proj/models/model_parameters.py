MANUAL_PARAMS = {
    "RandomForest": {
         'n_estimators': 92,
         'max_depth': 19,
         'min_samples_split': 10,
         'min_samples_leaf': 1,
         'bootstrap': True,
         'max_leaf_nodes': 58
    },
    "LightGBM": {
        'n_estimators': 395,
        'learning_rate': 0.023039879310004143,
        'max_depth': 7,
        'min_child_samples': 30,
        'subsample': 0.743738653112448,
        'colsample_bytree': 0.8352067066995073,
        'reg_alpha': 0.632171864299299,
        'reg_lambda': 0.09822796931465623
    },
    "XGBoost": {
        'n_estimators': 265,
        'learning_rate': 0.05887131322721688,
        'max_depth': 10,
        'min_child_weight': 10,
        'gamma': 2.029023552289681,
        'subsample': 0.6227921412463687,
        'colsample_bytree': 0.6186471014152328,
        'reg_alpha': 0.8403666683553538,
        'reg_lambda': 0.01397404059963454
    },
}

RFR_PARAMS = {'n_estimators': 81, 'max_depth': 13, 'min_samples_split': 10, 'min_samples_leaf': 1, 'bootstrap': True, 'max_leaf_nodes': 63}

LGBM_PARAMS = {'n_estimators': 146, 'learning_rate': 0.1894163467997181, 'max_depth': 10, 'min_child_samples': 7, 'subsample': 0.6012940387570913, 'colsample_bytree': 0.7307865541286798, 'reg_alpha': 0.8960024018173693, 'reg_lambda': 0.19625297013684237}

XGB_PARAMS = {'n_estimators': 229, 'learning_rate': 0.19844298446228187, 'max_depth': 7, 'min_child_weight': 6, 'gamma': 2.653413357911681, 'subsample': 0.6061972977699855, 'colsample_bytree': 0.8364115718996126, 'reg_alpha': 0.8616111211196239, 'reg_lambda': 0.35457994999219394}
