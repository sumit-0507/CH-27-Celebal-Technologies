"""Train and evaluate a LightGBM model on engineered demand features."""

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import lightgbm as lgb
except ModuleNotFoundError as error:
    if error.name == "lightgbm":
        raise SystemExit(
            "LightGBM is not installed in this Python environment. "
            "Install it with: python -m pip install lightgbm"
        ) from error
    raise


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_COLUMN = "OrderVolume"
IDENTIFIER_COLUMNS = {"Date", "Id"}
CATEGORICAL_COLUMNS = {
    "HubID",
    "HubFormat",
    "AssortmentTier",
    "LoyaltyProgram",
    "LoyaltyProgramInterval",
}


def update_status(
    status_path: Path,
    started_at: datetime,
    started_clock: float,
    state: str,
    message: str,
    **details: Any,
) -> None:
    status = {
        "state": state,
        "message": message,
        "pid": os.getpid(),
        "started_at": started_at.isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started_clock, 1),
        **details,
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = status_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    temporary_path.replace(status_path)


def rmsle(actual: pd.Series, predicted: pd.Series) -> float:
    squared_log_errors = [
        (
            math.log1p(float(prediction))
            - math.log1p(float(target))
        )
        ** 2
        for target, prediction in zip(actual, predicted)
    ]
    return math.sqrt(sum(squared_log_errors) / len(actual))


def prepare_features(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    excluded = IDENTIFIER_COLUMNS | {TARGET_COLUMN}
    feature_columns = [
        column
        for column in training.columns
        if column not in excluded and column in evaluation.columns
    ]
    if not feature_columns:
        raise ValueError("No shared model feature columns were found.")

    missing_from_training = set(evaluation.columns) - excluded - set(
        training.columns
    )
    if missing_from_training:
        raise ValueError(
            "Evaluation data contains features missing from training data: "
            f"{', '.join(sorted(missing_from_training))}"
        )

    x_train = training[feature_columns].copy()
    x_evaluation = evaluation[feature_columns].copy()
    categorical_columns = [
        column for column in CATEGORICAL_COLUMNS if column in feature_columns
    ]

    for column in categorical_columns:
        categories = pd.Index(
            pd.concat(
                [
                    x_train[column].astype("string"),
                    x_evaluation[column].astype("string"),
                ],
                ignore_index=True,
            ).dropna().unique()
        )
        x_train[column] = pd.Categorical(
            x_train[column].astype("string"),
            categories=categories,
        )
        x_evaluation[column] = pd.Categorical(
            x_evaluation[column].astype("string"),
            categories=categories,
        )

    for column in feature_columns:
        if column not in categorical_columns:
            x_train[column] = pd.to_numeric(x_train[column], errors="raise")
            x_evaluation[column] = pd.to_numeric(
                x_evaluation[column],
                errors="raise",
            )

    return x_train, x_evaluation, categorical_columns


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit LightGBM to forecast-style training features and score the "
            "chronological evaluation period using RMSLE."
        )
    )
    parser.add_argument(
        "--train-features",
        type=Path,
        default=PROJECT_ROOT / "data" / "orders_train_features.csv",
        help="Engineered training feature CSV.",
    )
    parser.add_argument(
        "--eval-features",
        type=Path,
        default=PROJECT_ROOT / "data" / "orders_eval_features.csv",
        help="Engineered evaluation feature CSV with target values.",
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "orders_eval_predictions.csv",
        help="CSV path for evaluation predictions.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "lightgbm_order_volume.txt",
        help="Path for the saved LightGBM model.",
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
        default=25,
        help="Print and save progress every N boosting iterations (default: 25).",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "lightgbm_training_status.json",
        help="Path for the live training status JSON file.",
    )
    args = parser.parse_args()

    if args.estimators < 1:
        parser.error("--estimators must be at least 1.")
    if args.progress_every < 1:
        parser.error("--progress-every must be at least 1.")

    started_at = datetime.now(timezone.utc)
    started_clock = time.monotonic()
    update_status(
        args.status_file,
        started_at,
        started_clock,
        "loading_data",
        "Reading engineered training and evaluation CSV files.",
    )
    print("[status] Reading engineered feature files...", flush=True)
    training = pd.read_csv(args.train_features)
    evaluation = pd.read_csv(args.eval_features)
    required_columns = {TARGET_COLUMN, "IsOpen"}
    for name, frame in (("training", training), ("evaluation", evaluation)):
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{name} features are missing required columns: "
                f"{', '.join(sorted(missing))}"
            )

    training = training.loc[training["IsOpen"].eq(1)].copy()
    if training.empty:
        raise ValueError("No open-hub training rows remain.")
    if evaluation.empty:
        raise ValueError("Evaluation feature file contains no rows.")

    update_status(
        args.status_file,
        started_at,
        started_clock,
        "preparing_features",
        f"Preparing {len(training):,} training and "
        f"{len(evaluation):,} evaluation rows.",
        training_rows=len(training),
        evaluation_rows=len(evaluation),
    )
    print(
        f"[status] Preparing {len(training):,} training rows and "
        f"{len(evaluation):,} evaluation rows...",
        flush=True,
    )
    y_train = pd.to_numeric(training[TARGET_COLUMN], errors="raise")
    y_evaluation = pd.to_numeric(evaluation[TARGET_COLUMN], errors="raise")
    if (y_train < 0).any() or (y_evaluation < 0).any():
        raise ValueError("OrderVolume targets must be non-negative.")
    if not y_train.map(math.isfinite).all() or not y_evaluation.map(
        math.isfinite
    ).all():
        raise ValueError("OrderVolume targets must be finite.")

    x_train, x_evaluation, categorical_columns = prepare_features(
        training,
        evaluation,
    )
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=args.estimators,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    def report_training_progress(environment: Any) -> None:
        iteration = environment.iteration + 1
        if (
            iteration == 1
            or iteration % args.progress_every == 0
            or iteration == args.estimators
        ):
            message = (
                f"LightGBM training iteration {iteration}/"
                f"{args.estimators}."
            )
            print(f"[progress] {message}", flush=True)
            update_status(
                args.status_file,
                started_at,
                started_clock,
                "training",
                message,
                current_iteration=iteration,
                total_iterations=args.estimators,
                training_rows=len(training),
            )

    report_training_progress.order = 10
    report_training_progress.before_iteration = False
    update_status(
        args.status_file,
        started_at,
        started_clock,
        "training",
        f"Starting LightGBM training for {args.estimators} iterations.",
        current_iteration=0,
        total_iterations=args.estimators,
        training_rows=len(training),
    )
    print(
        f"[status] Starting LightGBM training for "
        f"{args.estimators} iterations...",
        flush=True,
    )
    try:
        model.fit(
            x_train,
            y_train.map(math.log1p),
            categorical_feature=categorical_columns,
            callbacks=[report_training_progress],
        )
    except Exception as error:
        update_status(
            args.status_file,
            started_at,
            started_clock,
            "failed",
            f"{type(error).__name__}: {error}",
        )
        raise

    update_status(
        args.status_file,
        started_at,
        started_clock,
        "predicting",
        "Training finished; predicting evaluation rows.",
        current_iteration=args.estimators,
        total_iterations=args.estimators,
    )
    print("[status] Training finished; predicting evaluation rows...", flush=True)
    predictions = pd.Series(0.0, index=evaluation.index, name="PredictedOrderVolume")
    open_evaluation = evaluation["IsOpen"].eq(1)
    if open_evaluation.any():
        log_predictions = model.predict(
            x_evaluation.loc[open_evaluation],
        )
        predictions.loc[open_evaluation] = [
            max(0.0, math.expm1(float(value))) for value in log_predictions
        ]

    score = rmsle(y_evaluation, predictions)
    print(f"Training rows (open hubs): {len(training):,}")
    print(f"Evaluation rows: {len(evaluation):,}")
    print(f"Evaluation RMSLE: {score:.6f}")

    update_status(
        args.status_file,
        started_at,
        started_clock,
        "saving_outputs",
        f"Saving outputs; evaluation RMSLE is {score:.6f}.",
        evaluation_rmsle=score,
    )
    args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    prediction_columns = [
        column
        for column in ("Id", "HubID", "Date", "IsOpen")
        if column in evaluation.columns
    ]
    output = evaluation[prediction_columns].copy()
    output["ActualOrderVolume"] = y_evaluation
    output["PredictedOrderVolume"] = predictions
    output.to_csv(args.prediction_output, index=False)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(args.model_output))
    print(f"Predictions saved to {args.prediction_output}")
    print(f"Model saved to {args.model_output}")
    update_status(
        args.status_file,
        started_at,
        started_clock,
        "completed",
        f"Training and evaluation completed with RMSLE {score:.6f}.",
        current_iteration=args.estimators,
        total_iterations=args.estimators,
        evaluation_rmsle=score,
        prediction_output=str(args.prediction_output),
        model_output=str(args.model_output),
    )
    print(f"Status file: {args.status_file}")


if __name__ == "__main__":
    main()
