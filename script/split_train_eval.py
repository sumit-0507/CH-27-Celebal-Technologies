"""Split the latest dates from the training CSV using pandas."""

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def split_train_eval(
    input_path: Path,
    output_dir: Path,
    evaluation_days: int = 42,
) -> tuple[Path, Path]:
    if evaluation_days < 1:
        raise ValueError("evaluation_days must be at least 1.")

    data = pd.read_csv(input_path, parse_dates=["Date"])
    required_columns = {"HubID", "Date", "OrderVolume"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing_columns))}"
        )
    if data["Date"].isna().any():
        raise ValueError("Date contains missing or invalid values.")
    if data[["HubID", "Date"]].isna().any().any():
        raise ValueError("HubID and Date must not contain missing values.")
    if data.duplicated(["HubID", "Date"]).any():
        raise ValueError("Duplicate HubID/Date rows were found.")

    dates = data["Date"].drop_duplicates().sort_values()
    if len(dates) <= evaluation_days:
        raise ValueError(
            f"Need more than {evaluation_days} distinct dates; found {len(dates)}."
        )

    evaluation_dates = dates.iloc[-evaluation_days:]
    expected_dates = pd.date_range(
        start=evaluation_dates.iloc[0],
        end=evaluation_dates.iloc[-1],
        freq="D",
    )
    if not evaluation_dates.reset_index(drop=True).equals(
        pd.Series(expected_dates)
    ):
        raise ValueError(
            f"The latest {evaluation_days} distinct dates are not "
            "contiguous calendar days."
        )

    evaluation_start = evaluation_dates.iloc[0]
    training = data.loc[data["Date"] < evaluation_start]
    evaluation = data.loc[data["Date"] >= evaluation_start]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "orders_train_history.csv"
    evaluation_path = output_dir / "orders_eval.csv"
    training.to_csv(train_path, index=False, date_format="%Y-%m-%d")
    evaluation.to_csv(evaluation_path, index=False, date_format="%Y-%m-%d")

    print(
        f"Training history: {len(training):,} rows "
        f"({training['Date'].min():%Y-%m-%d} to "
        f"{training['Date'].max():%Y-%m-%d}) -> {train_path}"
    )
    print(
        f"Evaluation set: {len(evaluation):,} rows "
        f"({evaluation['Date'].min():%Y-%m-%d} to "
        f"{evaluation['Date'].max():%Y-%m-%d}) -> {evaluation_path}"
    )
    return train_path, evaluation_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split the latest contiguous dates into an evaluation set."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "orders_train.csv",
        help="Input training CSV (default: data/orders_train.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory for the split CSVs (default: data).",
    )
    parser.add_argument(
        "--evaluation-days",
        type=int,
        default=42,
        help="Number of latest contiguous dates for evaluation (default: 42).",
    )
    args = parser.parse_args()
    split_train_eval(args.input, args.output_dir, args.evaluation_days)


if __name__ == "__main__":
    main()
