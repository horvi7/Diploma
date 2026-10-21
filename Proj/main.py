def run_pipeline() -> None:
    import compare_metrics
    import create_gold_db
    import data_collector
    import db_handler
    import dice
    import predict_and_visualize
    import profiling
    import train_models_with_optuna
    import what_if_simulation

    #data_collector.collect_datasets()
    #db_handler.refresh_database()
    create_gold_db.create_gold_tables()

    co2_df = db_handler.get_from_db("gold_co2")
    if co2_df is None:
        return
    features = co2_df.drop(columns="co2")
    target = co2_df["co2"]
    features = features[["energy_per_capita", "primary_energy_consumption", "log_population", "log_gdp"]]
    #train_models_with_optuna.compare_manual_and_optuna(features, target)

    #compare_metrics.make_metrics(features, target)

    #predict_and_visualize.predict_and_visualize_regressor(features, target)
    #predict_and_visualize.predict_and_visualize_time_series()
    #predict_and_visualize.print_surrogate_tree(features, target)
    #dice.analyze_decision(co2_df, features)

    what_if_simulation.predict_and_visualize()
    # profiling.run_profiling()


if __name__ == "__main__":
    run_pipeline()