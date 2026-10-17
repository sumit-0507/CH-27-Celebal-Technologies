"""Build leakage-aware, multi-day LightGBM features with pandas."""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAG_DAYS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 28)
KNOWN_FUTURE_COLUMNS = (
    "IsOpen",
    "PromoActive",
    "RegionalHoliday",
    "SchoolClosureFlag",
)


def validate_frame(
    frame: pd.DataFrame,
    name: str,
    required_columns: set[str],
) -> None:
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{name} is missing columns: {', '.join(sorted(missing))}"
        )
    if frame["HubID"].isna().any():
        raise ValueError(f"{name} contains missing HubID values.")
    if frame[["HubID", "Date"]].duplicated().any():
        raise ValueError(f"{name} contains duplicate HubID/Date rows.")
    if frame["Date"].isna().any():
        raise ValueError(f"{name} contains missing or invalid dates.")

    if "IsOpen" in frame.columns and "OrderVolume" in frame.columns:
        order_volume = pd.to_numeric(frame["OrderVolume"], errors="coerce")
        invalid_target = order_volume.isna() | (
            frame["IsOpen"].eq(0) & order_volume.ne(0)
        )
        if invalid_target.any():
            raise ValueError(
                f"{name} has missing/non-numeric OrderVolume values or "
                "closed-hub rows whose OrderVolume is not zero."
            )


def add_calendar_features(
    frame: pd.DataFrame,
    first_date: pd.Timestamp,
) -> pd.DataFrame:
    result = frame.copy()
    dates = result["Date"]
    result["Year"] = dates.dt.year.astype("int16")
    result["Month"] = dates.dt.month.astype("int8")
    result["DayOfMonth"] = dates.dt.day.astype("int8")
    result["DayOfWeek"] = dates.dt.dayofweek.astype("int8")
    result["DayOfYear"] = dates.dt.dayofyear.astype("int16")
    result["WeekOfYear"] = dates.dt.isocalendar().week.astype("int8")
    result["IsWeekend"] = dates.dt.dayofweek.isin([5, 6]).astype("int8")
    result["DaysSinceStart"] = (dates - first_date).dt.days.astype("int32")
    return result


def latest_per_hub(
    ordered_history: pd.DataFrame,
    values: pd.Series,
    name: str,
) -> pd.DataFrame:
    latest = ordered_history[["HubID"]].copy()
    latest[name] = values
    return latest.groupby("HubID", sort=False).tail(1)


def make_origin_demand_features(history: pd.DataFrame) -> pd.DataFrame:
    """Summarize demand at a forecast origin, averaging open days only."""
    ordered = history.sort_values(["HubID", "Date"]).copy()
    demand = ordered.groupby("HubID", sort=False)["OrderVolume"]
    open_demand = ordered["OrderVolume"].where(ordered["IsOpen"].eq(1))
    open_demand_by_hub = open_demand.groupby(ordered["HubID"], sort=False)
    features = pd.DataFrame({"HubID": ordered["HubID"].drop_duplicates()})

    for lag in LAG_DAYS:
        # The latest history row is the forecast origin. Lag 1 is its target.
        lagged_values = demand.shift(lag - 1)
        features = features.merge(
            latest_per_hub(ordered, lagged_values, f"Lag{lag}"),
            on="HubID",
            how="left",
            validate="one_to_one",
        )

    for window in ROLLING_WINDOWS:
        rolling_mean = open_demand_by_hub.transform(
            lambda values: values.rolling(window, min_periods=1).mean()
        )
        rolling_std = open_demand_by_hub.transform(
            lambda values: values.rolling(window, min_periods=2).std()
        )
        zero_rate = open_demand_by_hub.transform(
            lambda values: values.eq(0)
            .where(values.notna())
            .rolling(window, min_periods=1)
            .mean()
        )
        window_features = latest_per_hub(
            ordered, rolling_mean, f"RollingMean{window}"
        ).merge(
            latest_per_hub(ordered, rolling_std, f"RollingStd{window}"),
            on="HubID",
            validate="one_to_one",
        ).merge(
            latest_per_hub(ordered, zero_rate, f"ZeroRate{window}"),
            on="HubID",
            validate="one_to_one",
        )
        features = features.merge(
            window_features,
            on="HubID",
            how="left",
            validate="one_to_one",
        )

    expanding_mean = open_demand_by_hub.transform(
        lambda values: values.expanding(min_periods=1).mean()
    )
    features = features.merge(
        latest_per_hub(ordered, expanding_mean, "HubExpandingMean"),
        on="HubID",
        how="left",
        validate="one_to_one",
    )
    return features


def build_forecast_features(
    history: pd.DataFrame,
    future_rows: pd.DataFrame,
    metadata: pd.DataFrame,
    first_date: pd.Timestamp,
    forecast_origin: pd.Timestamp,
) -> pd.DataFrame:
    result = add_calendar_features(future_rows, first_date)
    result = result.merge(
        metadata,
        on="HubID",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if result["_merge"].ne("both").any():
        missing_hubs = result.loc[
            result["_merge"].ne("both"), "HubID"
        ].drop_duplicates()
        raise ValueError(
            f"Metadata is missing for HubID values: "
            f"{missing_hubs.head(10).tolist()}"
        )
    result = result.drop(columns="_merge")

    origin_features = make_origin_demand_features(history)
    result = result.merge(
        origin_features,
        on="HubID",
        how="left",
        validate="many_to_one",
    )

    weekday_history = history.loc[history["IsOpen"].eq(1)].assign(
        DayOfWeek=lambda rows: rows["Date"].dt.dayofweek
    )
    weekday_means = weekday_history.groupby(
        ["HubID", "DayOfWeek"],
        as_index=False,
    )["OrderVolume"].mean().rename(
        columns={"OrderVolume": "HubWeekdayMean"}
    )
    result = result.merge(
        weekday_means,
        on=["HubID", "DayOfWeek"],
        how="left",
        validate="many_to_one",
    )
    result["ForecastHorizonDays"] = (
        result["Date"] - forecast_origin
    ).dt.days.astype("int16")

    for column in KNOWN_FUTURE_COLUMNS:
        if column not in result.columns:
            raise ValueError(
                f"Forecast rows are missing known feature {column!r}."
            )

    # These columns are unavailable for the competition test period or are
    # redundant with derived calendar features.
    result = result.drop(
        columns=["AppSessions", "Weekday"],
        errors="ignore",
    )
    return result


def build_training_features(
    history: pd.DataFrame,
    metadata: pd.DataFrame,
    forecast_days: int,
    update_status: Callable[..., None],
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(history["Date"].drop_duplicates().sort_values())
    if len(dates) <= forecast_days:
        raise ValueError(
            f"Need more than {forecast_days} history dates to create "
            "forecast-style training rows."
        )

    expected_dates = pd.date_range(dates[0], dates[-1], freq="D")
    if not dates.equals(expected_dates):
        raise ValueError("History dates must be contiguous calendar days.")

    blocks = []
    block_starts = list(range(forecast_days, len(dates), forecast_days))
    total_blocks = len(block_starts)
    # Each block shares one frozen origin snapshot, mirroring a multi-day
    # forecast where actual demand is not known during the horizon.
    for block_number, start in enumerate(block_starts, start=1):
        block_dates = dates[start : start + forecast_days]
        if len(block_dates) == 0:
            continue

        forecast_origin = dates[start - 1]
        block_end = block_dates[-1]
        update_status(
            "building_training_features",
            (
                f"Building training block {block_number}/{total_blocks}: "
                f"{block_dates[0]:%Y-%m-%d} to {block_end:%Y-%m-%d}"
            ),
            completed_blocks=block_number - 1,
            total_blocks=total_blocks,
        )
        print(
            f"[{block_number}/{total_blocks}] Building training features "
            f"for {block_dates[0]:%Y-%m-%d} to {block_end:%Y-%m-%d}...",
            flush=True,
        )
        context = history.loc[history["Date"] <= forecast_origin]
        block_rows = history.loc[history["Date"].isin(block_dates)]
        block = build_forecast_features(
            history=context,
            future_rows=block_rows,
            metadata=metadata,
            first_date=dates[0],
            forecast_origin=forecast_origin,
        )
        blocks.append(block)
        update_status(
            "building_training_features",
            (
                f"Finished training block {block_number}/{total_blocks}: "
                f"{len(block):,} rows"
            ),
            completed_blocks=block_number,
            total_blocks=total_blocks,
        )
        print(
            f"[{block_number}/{total_blocks}] Finished "
            f"({len(block):,} rows).",
            flush=True,
        )

    features = pd.concat(blocks, ignore_index=True)
    # Closed rows remain in history for correct date offsets and zero-demand
    # statistics, but are not useful supervised training examples.
    features = features.loc[features["IsOpen"].eq(1)]
    return features.sort_values(["Date", "HubID"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create chronological LightGBM features using pandas. "
            "Training samples use frozen history snapshots by forecast block."
        )
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=PROJECT_ROOT / "data" / "orders_train_history.csv",
        help="Historical training CSV.",
    )
    parser.add_argument(
        "--future",
        type=Path,
        default=PROJECT_ROOT / "data" / "orders_eval.csv",
        help="Future rows to engineer (evaluation or test CSV).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data" / "hub_metadata.csv",
        help="Hub metadata CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory for engineered CSV files.",
    )
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=42,
        help="Forecast horizon and training block size (default: 42).",
    )
    args = parser.parse_args()
    if args.forecast_days < 1:
        parser.error("--forecast-days must be at least 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "feature_engineering_status.json"
    started_at = datetime.now(timezone.utc)
    started_clock = time.monotonic()

    def update_status(
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
        temporary_path = status_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(status, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(status_path)
        print(f"[status] {message}", flush=True)

    try:
        update_status("running", "Starting feature engineering.")
        update_status("loading_data", "Reading history, future, and metadata CSVs.")
        history = pd.read_csv(args.history, parse_dates=["Date"])
        future = pd.read_csv(args.future, parse_dates=["Date"])
        metadata = pd.read_csv(args.metadata)
        update_status(
            "validating_data",
            (
                f"Loaded {len(history):,} history rows, "
                f"{len(future):,} future rows, and "
                f"{len(metadata):,} metadata rows."
            ),
        )

        validate_frame(
            history,
            "history",
            {"HubID", "Date", "OrderVolume", "IsOpen"},
        )
        validate_frame(future, "future", {"HubID", "Date", "IsOpen"})
        missing_metadata_columns = {"HubID"}.difference(metadata.columns)
        if missing_metadata_columns:
            raise ValueError(
                "metadata is missing columns: "
                f"{', '.join(sorted(missing_metadata_columns))}"
            )
        if metadata["HubID"].isna().any():
            raise ValueError("metadata contains missing HubID values.")
        if metadata["HubID"].duplicated().any():
            raise ValueError("Metadata contains duplicate HubID values.")

        history = history.sort_values(["Date", "HubID"]).reset_index(drop=True)
        future = future.sort_values(["Date", "HubID"]).reset_index(drop=True)
        if future["Date"].min() <= history["Date"].max():
            raise ValueError(
                "Every future date must be after the final history date."
            )
        future_dates = pd.DatetimeIndex(
            future["Date"].drop_duplicates().sort_values()
        )
        if len(future_dates) > args.forecast_days:
            raise ValueError(
                f"Future data has {len(future_dates)} dates, exceeding the "
                f"{args.forecast_days}-day forecast horizon."
            )
        if not future_dates.equals(
            pd.date_range(future_dates[0], future_dates[-1], freq="D")
        ):
            raise ValueError("Future dates must be contiguous calendar days.")

        training_path = args.output_dir / "orders_train_features.csv"
        future_path = args.output_dir / f"{args.future.stem}_features.csv"
        update_status(
            "building_training_features",
            "Starting chronological training feature generation.",
        )
        training_features = build_training_features(
            history,
            metadata,
            args.forecast_days,
            update_status,
        )
        update_status(
            "building_future_features",
            "Building evaluation/test features from historical data.",
        )
        future_features = build_forecast_features(
            history=history,
            future_rows=future,
            metadata=metadata,
            first_date=history["Date"].min(),
            forecast_origin=history["Date"].max(),
        )

        update_status("writing_outputs", "Writing engineered feature CSV files.")
        training_features.to_csv(
            training_path,
            index=False,
            date_format="%Y-%m-%d",
        )
        future_features.to_csv(
            future_path,
            index=False,
            date_format="%Y-%m-%d",
        )
        update_status(
            "completed",
            (
                f"Completed. Wrote {len(training_features):,} training rows "
                f"and {len(future_features):,} future rows."
            ),
            training_rows=len(training_features),
            future_rows=len(future_features),
            training_output=str(training_path),
            future_output=str(future_path),
        )
        print(f"Training features: {len(training_features):,} rows -> {training_path}")
        print(f"Future features: {len(future_features):,} rows -> {future_path}")
        print(f"Status file: {status_path}")
    except Exception as error:
        update_status(
            "failed",
            f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()
