from analysis.compare_metrics import make_metrics
from analysis.dice import analyze_decision
from analysis.what_if_simulation import predict_and_visualize as predict_what_if
from data.data_collector import collect_datasets
from data.db_handler import get_from_db
from models.train_models_with_optuna import compare_manual_and_optuna
from visualization.predict_and_visualize import predict_and_visualize
from visualization.profiling import run_profiling


def run_pipeline() -> None:
    collect_datasets()
    co2_df = get_from_db("gold_co2")
    if co2_df is None:
        return
    features = co2_df.drop(columns="co2")
    target = co2_df["co2"]

    compare_manual_and_optuna(features, target)
    make_metrics(features, target)

    predict_and_visualize(features, target)
    analyze_decision(co2_df, features, target)

    predict_what_if(co2_df)
    run_profiling()


if __name__ == "__main__":
    run_pipeline()