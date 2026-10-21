import dice_ml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import db_handler as db
import trained_models as tm

#Itt megfelelő featureök kellenek a modellben is. Kisérletezni lehet. Kusnetzet a min és max követi
def analyze_decision(df, features):
    df = db.get_from_db("gold_co2")
    df = df[
        ["co2", "log_gdp_per_population"]
    ].reset_index(drop=True)
    features = df.drop(columns="co2")

    dataset = dice_ml.Data(
        dataframe=df,
        continuous_features=features.columns.tolist(),
        outcome_name="co2",
    )
    model = tm.get_model("XGBoost")
    dice_model = dice_ml.Model(
        model=model,
        backend="sklearn",
        model_type="regressor",
    )

    explanation = dice_ml.Dice(dataset, dice_model)
    selected_rows = {
        "Biggest GDP": df["log_gdp_per_population"].idxmax(),
        "Lowest GDP": df["log_gdp_per_population"].idxmin()
    }

    for label, row_index in selected_rows.items():
        feature_row_to_explain = features.iloc[[row_index]]
        prediction = model.predict(feature_row_to_explain)
        counterfactual = explanation.generate_counterfactuals(
            feature_row_to_explain,
            total_CFs=1,
            desired_range=[prediction[0] * 1.01, prediction[0] * 1.15],
            features_to_vary=features.columns.tolist(),
        )

        changed_features = counterfactual.cf_examples_list[0].final_cfs_df
        comparison = pd.concat(
            [
                feature_row_to_explain.assign(version="Original"),
                changed_features.assign(version="Changed"),
            ],
            ignore_index=True,
        )
        print(
            f"\n{label} (GDP: {df.loc[row_index, 'log_gdp_per_population']}, "
            f"CO2: {df.loc[row_index, 'co2']}):"
        )
        print(comparison[["version"] + features.columns.tolist()])

if __name__ == "__main__":
  analyze_decision()