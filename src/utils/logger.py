import csv
from datetime import datetime
from pathlib import Path


class TrainingLogger:
    def __init__(self, log_dir="outputs/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"train_{timestamp}.csv"
        self._initialized = False

    def log(self, epoch: int, metrics: dict):
        fieldnames = ["epoch"] + list(metrics.keys())

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not self._initialized:
                writer.writeheader()
                self._initialized = True

            writer.writerow({"epoch": epoch, **metrics})

    def get_log_path(self) -> str:
        return str(self.log_path)
