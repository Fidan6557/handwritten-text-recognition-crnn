import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(log_path, output_dir="outputs/plots"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Training Metrics", fontsize=14)

    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss")
    axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("CTC Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df["epoch"], df["cer"], color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("CER")
    axes[1].set_title("Character Error Rate")
    axes[1].grid(True)

    axes[2].plot(df["epoch"], df["wer"], color="red")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("WER")
    axes[2].set_title("Word Error Rate")
    axes[2].grid(True)

    plt.tight_layout()

    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Plot saved to: {output_path}")


def _find_latest_log(log_dir="outputs/logs"):
    log_dir = Path(log_dir)
    log_files = sorted(log_dir.glob("train_*.csv"))

    if not log_files:
        raise FileNotFoundError(f"No training log files found in {log_dir}")

    return log_files[-1]


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from a CSV log.")
    parser.add_argument("--log", default=None, help="Path to training log CSV (default: latest in outputs/logs/).")
    parser.add_argument("--output-dir", default="outputs/plots", help="Directory to save plots.")
    args = parser.parse_args()

    log_path = args.log if args.log else _find_latest_log()
    plot_training_curves(log_path, args.output_dir)


if __name__ == "__main__":
    main()
