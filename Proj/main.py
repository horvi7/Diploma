def run_pipeline() -> None:
    import compare_metrics
    import create_gold_db
    import data_collector
    import db_handler
    import dice
    import predict_and_visualize
    import profiling
    import what_if_simulation

    data_collector.collect_datasets()
    db_handler.refresh_database()
    create_gold_db.create_gold_tables()

    co2_df = db_handler.get_from_db("gold_co2")
    if co2_df is None:
        return
    compare_metrics.make_metrics(
        co2_df.drop(columns="co2"),
        co2_df["co2"],
    )

    predict_and_visualize.predict_and_visualize()
    time_series_df = db_handler.get_from_db("gold_co2_time_series")
    if time_series_df is None:
        print("Skipping time-series analysis: gold_co2_time_series is unavailable.")
    else:
        predict_and_visualize.predict_and_visualize_time_series()
    predict_and_visualize.print_surrogate_tree()
    dice.analyze_decision()
    what_if_simulation.predict_and_visualize()
    profiling.run_profiling()


if __name__ == "__main__":
    run_pipeline()