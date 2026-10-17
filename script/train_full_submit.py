"""Train on all available history and create the competition submission."""

import argparse
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from train_lightgbm import (
    PROJECT_ROOT,
    TARGET_COLUMN,
    prepare_features,
    update_status,
)

try:
    import lightgbm as lgb
except ModuleNotFoundError as error:
    if error.name == "lightgbm":
        raise SystemExit(
            "LightGBM is not installed in this Python environment. "
            "Install it with: python -m pip install lightgbm"
        ) from error
    raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train LightGBM on all engineered training rows and write "
            "predictions in the sample-submission format."
        )
    )
    parser.add_argument(
        "--train-features",
        type=Path,
        default=PROJECT_ROOT / "data" / "orders_train_features.csv",
        help="Features generated from the full orders_train.csv.",
    )
    parser.add_argument(
        "--test-features",
        type=Path,
        default=PROJECT_ROOT / "data" / "orders_test_features.csv",
        help="Features generated from orders_test.csv.",
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=PROJECT_ROOT / "data" / "sample_submission.csv",
        help="Competition sample submission used to validate ID order.",
    )
    parser.add_argument(
        "--submission-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "submission.csv",
        help="Output competition submission CSV.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "lightgbm_order_volume_full.txt",
        help="Path for the model trained on all history.",
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
        help="Print progress every N iterations (default: 25).",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "lightgbm_training_status.json",
        help="Path for live training status.",
    )
    args = parser.parse_args()
    if args.estimators < 1 or args.progress_every < 1:
        parser.error("--estimators and --progress-every must be positive.")

    started_at = datetime.now(timezone.utc)
    started_clock = time.monotonic()
    update_status(
        args.status_file,
        started_at,
        started_clock,
        "loading_data",
        "Reading full-history features, test features, and sample submission.",
    )
    print("[status] Reading feature and sample-submission files...", flush=True)
    training = pd.read_csv(args.train_features)
    test = pd.read_csv(args.test_features)
    sample = pd.read_csv(args.sample_submission)

    for name, frame, columns in (
        ("training", training, {TARGET_COLUMN, "IsOpen"}),
        ("test", test, {"IsOpen", "Id"}),
        ("sample submission", sample, {"Id", TARGET_COLUMN}),
    ):
        missing = columns.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{name} is missing required columns: "
                f"{', '.join(sorted(missing))}"
            )

    if sample.columns.tolist() != ["Id", TARGET_COLUMN]:
        raise ValueError(
            "The sample submission must contain exactly the columns "
            "'Id,OrderVolume' in that order."
        )
    if sample["Id"].isna().any() or sample["Id"].duplicated().any():
        raise ValueError("Sample submission IDs must be present and unique.")
    if test["Id"].isna().any() or test["Id"].duplicated().any():
        raise ValueError("Test feature IDs must be present and unique.")
    if set(sample["Id"]) != set(test["Id"]):
        raise ValueError(
            "Test feature IDs do not match the sample-submission IDs."
        )

    training = training.loc[training["IsOpen"].eq(1)].copy()
    if training.empty:
        raise ValueError("No open-hub training rows remain.")
    targets = pd.to_numeric(training[TARGET_COLUMN], errors="raise")
    if (targets < 0).any() or not targets.map(math.isfinite).all():
        raise ValueError("Training targets must be finite and non-negative.")

    update_status(
        args.status_file,
        started_at,
        started_clock,
        "preparing_features",
        f"Preparing {len(training):,} open training rows and "
        f"{len(test):,} test rows.",
        training_rows=len(training),
        test_rows=len(test),
    )
    x_train, x_test, categorical_columns = prepare_features(training, test)

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
                f"Full-data LightGBM training iteration "
                f"{iteration}/{args.estimators}."
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
        f"Starting full-history training for {args.estimators} iterations.",
        current_iteration=0,
        total_iterations=args.estimators,
        training_rows=len(training),
    )
    print("[status] Training on all available history...", flush=True)
    try:
        model.fit(
            x_train,
            targets.map(math.log1p),
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
        "Full-history training finished; predicting test rows.",
        current_iteration=args.estimators,
        total_iterations=args.estimators,
    )
    predictions = pd.Series(0.0, index=test.index, name=TARGET_COLUMN)
    open_test = test["IsOpen"].eq(1)
    if open_test.any():
        log_predictions = model.predict(x_test.loc[open_test])
        predictions.loc[open_test] = [
            max(0.0, math.expm1(float(value))) for value in log_predictions
        ]

    prediction_by_id = pd.DataFrame(
        {"Id": test["Id"], TARGET_COLUMN: predictions}
    )
    submission = sample[["Id"]].merge(
        prediction_by_id,
        on="Id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if submission[TARGET_COLUMN].isna().any():
        raise ValueError("Some sample-submission IDs have no prediction.")
    submission[TARGET_COLUMN] = submission[TARGET_COLUMN].clip(lower=0)

    args.submission_output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission_output, index=False)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(args.model_output))
    update_status(
        args.status_file,
        started_at,
        started_clock,
        "completed",
        f"Submission created with {len(submission):,} predictions.",
        current_iteration=args.estimators,
        total_iterations=args.estimators,
        training_rows=len(training),
        submission_rows=len(submission),
        submission_output=str(args.submission_output),
        model_output=str(args.model_output),
    )
    print(f"Submission saved to {args.submission_output}")
    print(f"Full-history model saved to {args.model_output}")
    print(f"Status file: {args.status_file}")


if __name__ == "__main__":
    main()
