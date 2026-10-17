"""Report whether a tracked project job is currently running."""

import argparse
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TERMINAL_STATES = {"completed", "failed"}


def process_is_running(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        if getattr(error, "winerror", None) == 87:
            return False
        raise
    return True


def read_status(status_path: Path) -> dict[str, Any]:
    if not status_path.exists():
        return {
            "state": "not_started",
            "message": "No status file found. The job has not started, or it "
            "was started before status tracking was added.",
        }

    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid status file: {status_path}") from error

    state = status.get("state")
    if state in TERMINAL_STATES:
        status["is_running"] = False
        return status

    process_id = status.get("pid")
    if not isinstance(process_id, int):
        status["state"] = "unknown"
        status["is_running"] = False
        status["message"] = "Status file does not contain a valid process ID."
    else:
        status["is_running"] = process_is_running(process_id)
        if not status["is_running"]:
            status["state"] = "not_running"
            status["message"] = (
                "The recorded process is no longer running. It may have "
                "finished or stopped unexpectedly."
            )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check a tracked job's process status and latest progress."
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "feature_engineering_status.json",
        help=(
            "Status JSON to check (default: feature-engineering status). "
            "For LightGBM, use data/lightgbm_training_status.json."
        ),
    )
    args = parser.parse_args()

    status = read_status(args.status_file)
    print(f"State: {status.get('state', 'unknown')}")
    print(f"Running: {'yes' if status.get('is_running') else 'no'}")
    print(f"Progress: {status.get('message', 'No status message.')}")
    if "pid" in status:
        print(f"Process ID: {status['pid']}")
    if "updated_at" in status:
        print(f"Last update (UTC): {status['updated_at']}")
    if "elapsed_seconds" in status:
        print(f"Elapsed seconds: {status['elapsed_seconds']}")
    if "completed_blocks" in status and "total_blocks" in status:
        print(
            f"Training blocks: {status['completed_blocks']}/"
            f"{status['total_blocks']}"
        )
    if "current_iteration" in status and "total_iterations" in status:
        print(
            f"Training iterations: {status['current_iteration']}/"
            f"{status['total_iterations']}"
        )
    if "evaluation_rmsle" in status:
        print(f"Evaluation RMSLE: {status['evaluation_rmsle']:.6f}")


if __name__ == "__main__":
    main()
