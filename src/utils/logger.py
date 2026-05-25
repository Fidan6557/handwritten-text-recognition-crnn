import csv
from pathlib import Path
from datetime import datetime


class TrainingLogger:
    """
    Logs training metrics to a CSV file for later analysis.
    """

    def __init__(self, log_dir: str = "outputs/logs", run_name: str = None):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_path = log_dir / f"train_{run_name}.csv"
        self._initialized = False

    def log(self, epoch: int, metrics: dict):
        """
        Append a row of metrics to the CSV log.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metric names to values.
        """

        row = {"epoch": epoch, **metrics}

        if not self._initialized:
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)

            self._initialized = True

        else:
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writerow(row)

    def get_log_path(self) -> Path:
        return self.log_path
