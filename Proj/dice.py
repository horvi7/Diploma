import dice_ml

import db_handler as db
import trained_models as tm


def analyze_decision():
  df = db.get_from_db("gold_co2")
  print(df.head())

  features_kaya = df[[
      "log_population",
      "log_gdp_per_population",
      "log_energy_per_gdp",
      "log_co2_per_energy",
  ]]
  df_kaya = df[[
      "co2",
      "log_population",
      "log_gdp_per_population",
      "log_energy_per_gdp",
      "log_co2_per_energy",
  ]]

  features_all = df.drop(columns=["co2"])
  print("Features_all: ", features_all.columns.tolist())

  dice_dataset_kaya = dice_ml.Data(dataframe=df_kaya, continuous_features=features_kaya.columns.tolist(), outcome_name="co2")

  dice_dataset_all = dice_ml.Data(dataframe=df, continuous_features=features_all.columns.tolist(), outcome_name="co2")

  model = tm.get_model('RandomForest')

  dice_model = dice_ml.Model(model=model, backend="sklearn",model_type="regressor")

  exp = dice_ml.Dice(dice_dataset_kaya, dice_model)

  custom_feature = features_kaya.iloc[[0]]

  prediction = model.predict(custom_feature)
  print(f"Prediction for custom feature: {prediction}")

  decrease_range_low = 0.8
  decrease_range_high = 0.9

  cf = exp.generate_counterfactuals(custom_feature, total_CFs=1, desired_range=[prediction[0]*decrease_range_low , prediction[0]*decrease_range_high], features_to_vary=features_kaya.columns.tolist())

  cf_df = cf.cf_examples_list[0].final_cfs_df

  print("\nGenerált Counterfactual: ")
  print(cf_df)

  print("Összes feature-el: ")
  exp = dice_ml.Dice(dice_dataset_all, dice_model)
  custom_feature = features_all.iloc[[0]]
  cf_all = exp.generate_counterfactuals(custom_feature, total_CFs=1, desired_range=[prediction[0]*decrease_range_low , prediction[0]*decrease_range_high], features_to_vary=features_all.columns.tolist())
  cf_all_df = cf_all.cf_examples_list[0].final_cfs_df
  print(cf_all_df)

if __name__ == "__main__":
  analyze_decision()