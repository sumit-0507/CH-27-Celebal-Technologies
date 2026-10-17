"""Train XGBoost and CatBoost on the chronological evaluation split."""

import argparse
import math
from pathlib import Path

import pandas as pd

from train_lightgbm import PROJECT_ROOT, prepare_features


def rmsle(actual: pd.Series, predicted: pd.Series) -> float:
    errors = [
        (
            math.log1p(float(target))
            - math.log1p(max(0.0, float(prediction)))
        )
        ** 2
        for target, prediction in zip(actual, predicted)
    ]
    return math.sqrt(sum(errors) / len(errors))


def validate_targets(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if (numeric < 0).any() or not numeric.map(math.isfinite).all():
        raise ValueError(f"{name} targets must be finite and non-negative.")
    return numeric


def make_xgboost_progress_callback(xgboost, every: int, total: int):
    class ProgressCallback(xgboost.callback.TrainingCallback):
        def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
            iteration = epoch + 1
            if iteration == 1 or iteration % every == 0 or iteration == total:
                print(
                    f"[progress] XGBoost iteration {iteration}/{total}.",
                    flush=True,
                )
            return False

    return ProgressCallback()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit XGBoost and CatBoost on the same open-hub training rows and "
            "compare RMSLE on the chronological evaluation set."
        )
    )
    parser.add_argument(
        "--train-features",
        type=Path,
        default=PROJECT_ROOT / "data" / "orders_train_features.csv",
    )
    parser.add_argument(
        "--eval-features",
        type=Path,
        default=PROJECT_ROOT / "data" / "orders_eval_features.csv",
    )
    parser.add_argument(
        "--lightgbm-predictions",
        type=Path,
        default=PROJECT_ROOT / "data" / "orders_eval_predictions.csv",
        help="Existing LightGBM holdout predictions for comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default=800,
        help="Number of boosting iterations (default: 800).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Progress reporting interval for boosting iterations.",
    )
    args = parser.parse_args()
    if args.estimators < 1 or args.progress_every < 1:
        parser.error("--estimators and --progress-every must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import xgboost as xgb
    except ModuleNotFoundError as error:
        if error.name == "xgboost":
            raise SystemExit(
                "XGBoost is not installed in this environment. Install it "
                "with: .\\.venv\\Scripts\\python.exe -m pip install xgboost"
            ) from error
        raise

    try:
        from catboost import CatBoostRegressor
    except ModuleNotFoundError as error:
        if error.name == "catboost":
            raise SystemExit(
                "CatBoost is not installed in this environment. Install it "
                "with: .\\.venv\\Scripts\\python.exe -m pip install catboost"
            ) from error
        raise

    print("[status] Reading engineered training and evaluation files...", flush=True)
    training = pd.read_csv(args.train_features)
    evaluation = pd.read_csv(args.eval_features)
    required = {"OrderVolume", "IsOpen", "HubID", "Date"}
    for label, frame in (("training", training), ("evaluation", evaluation)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{label} data is missing columns: "
                f"{', '.join(sorted(missing))}"
            )
    if evaluation.empty:
        raise ValueError("Evaluation data is empty.")

    training = training.loc[training["IsOpen"].eq(1)].copy()
    if training.empty:
        raise ValueError("No open-hub training rows remain.")
    y_train = validate_targets(training["OrderVolume"], "Training")
    y_evaluation = validate_targets(evaluation["OrderVolume"], "Evaluation")
    x_train, x_evaluation, categorical_columns = prepare_features(
        training,
        evaluation,
    )
    y_train_log = y_train.map(math.log1p)
    open_evaluation = evaluation["IsOpen"].eq(1)
    print(
        f"[status] {len(training):,} open training rows; "
        f"{len(evaluation):,} evaluation rows; "
        f"{len(categorical_columns)} categorical features.",
        flush=True,
    )

    results = []
    predictions_by_model = {}

    print("[status] Training XGBoost...", flush=True)
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.estimators,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
        callbacks=[
            make_xgboost_progress_callback(
                xgb,
                args.progress_every,
                args.estimators,
            )
        ],
    )
    xgb_model.fit(
        x_train,
        y_train_log,
    )
    xgb_predictions = pd.Series(
        0.0,
        index=evaluation.index,
        name="PredictedOrderVolume",
    )
    if open_evaluation.any():
        xgb_log_predictions = xgb_model.predict(
            x_evaluation.loc[open_evaluation]
        )
        xgb_predictions.loc[open_evaluation] = [
            max(0.0, math.expm1(float(value)))
            for value in xgb_log_predictions
        ]
    xgb_score = rmsle(y_evaluation, xgb_predictions)
    predictions_by_model["XGBoost"] = xgb_predictions
    results.append({"Model": "XGBoost", "RMSLE": xgb_score})
    print(f"[result] XGBoost RMSLE: {xgb_score:.8f}", flush=True)
    xgb_model.save_model(str(args.output_dir / "xgboost_order_volume.json"))

    print("[status] Preparing CatBoost categorical features...", flush=True)
    cat_train = x_train.copy()
    cat_evaluation = x_evaluation.copy()
    for column in categorical_columns:
        cat_train[column] = cat_train[column].astype("string").fillna("__MISSING__")
        cat_evaluation[column] = (
            cat_evaluation[column].astype("string").fillna("__MISSING__")
        )
    cat_features = categorical_columns

    print("[status] Training CatBoost...", flush=True)
    cat_model = CatBoostRegressor(
        iterations=args.estimators,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        random_seed=42,
        thread_count=-1,
        verbose=args.progress_every,
        allow_writing_files=False,
    )
    cat_model.fit(
        cat_train,
        y_train_log,
        cat_features=cat_features,
    )
    cat_predictions = pd.Series(
        0.0,
        index=evaluation.index,
        name="PredictedOrderVolume",
    )
    if open_evaluation.any():
        cat_log_predictions = cat_model.predict(
            cat_evaluation.loc[open_evaluation]
        )
        cat_predictions.loc[open_evaluation] = [
            max(0.0, math.expm1(float(value)))
            for value in cat_log_predictions
        ]
    cat_score = rmsle(y_evaluation, cat_predictions)
    predictions_by_model["CatBoost"] = cat_predictions
    results.append({"Model": "CatBoost", "RMSLE": cat_score})
    print(f"[result] CatBoost RMSLE: {cat_score:.8f}", flush=True)
    cat_model.save_model(str(args.output_dir / "catboost_order_volume.cbm"))

    if args.lightgbm_predictions.exists():
        lightgbm = pd.read_csv(args.lightgbm_predictions)
        required_predictions = {
            "HubID",
            "Date",
            "PredictedOrderVolume",
        }
        missing = required_predictions.difference(lightgbm.columns)
        if missing:
            raise ValueError(
                "LightGBM predictions are missing columns: "
                f"{', '.join(sorted(missing))}"
            )
        keys = ["HubID", "Date"]
        if lightgbm.duplicated(keys).any():
            raise ValueError("LightGBM predictions have duplicate HubID/Date.")
        baseline = evaluation[keys].merge(
            lightgbm[keys + ["PredictedOrderVolume"]],
            on=keys,
            how="left",
            validate="one_to_one",
        )["PredictedOrderVolume"]
        if baseline.isna().any():
            raise ValueError(
                "LightGBM predictions do not cover every evaluation row."
            )
        baseline_score = rmsle(y_evaluation, baseline)
        results.insert(0, {"Model": "LightGBM", "RMSLE": baseline_score})
        predictions_by_model["LightGBM"] = baseline.reset_index(drop=True)
        print(f"[result] LightGBM RMSLE: {baseline_score:.8f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_output = evaluation[
        [column for column in ("Id", "HubID", "Date", "IsOpen") if column in evaluation]
    ].copy()
    for name, predictions in predictions_by_model.items():
        prediction_output[f"{name}Prediction"] = predictions.to_numpy()
    prediction_path = args.output_dir / "model_comparison_predictions.csv"
    prediction_output.to_csv(prediction_path, index=False)

    results_frame = pd.DataFrame(results).sort_values("RMSLE")
    results_path = args.output_dir / "model_comparison_rmsle.csv"
    results_frame.to_csv(results_path, index=False)
    print("\nRMSLE comparison (lower is better):")
    print(results_frame.to_string(index=False))
    print(f"Predictions: {prediction_path}")
    print(f"Scores: {results_path}")


if __name__ == "__main__":
    main()
