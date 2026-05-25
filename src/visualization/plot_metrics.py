from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(log_csv_path: str, output_dir: str = "outputs/plots"):
    """
    Plot training and validation loss, CER and WER from a training log CSV.

    Args:
        log_csv_path: Path to the CSV file produced by TrainingLogger.
        output_dir: Directory where the plot PNG will be saved.
    """

    log_csv_path = Path(log_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", marker="o")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CTC Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df["epoch"], df["cer"], label="CER", color="orange", marker="o")
    axes[1].set_title("Character Error Rate")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("CER")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(df["epoch"], df["wer"], label="WER", color="red", marker="o")
    axes[2].set_title("Word Error Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("WER")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()

    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {output_path}")

    return output_path


def plot_prediction_samples(predictions: list, targets: list, output_dir: str = "outputs/plots"):
    """
    Create a simple text comparison table of predictions vs ground truth.

    Args:
        predictions: List of predicted strings.
        targets: List of ground truth strings.
        output_dir: Directory to save the output figure.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = min(10, len(predictions))

    fig, ax = plt.subplots(figsize=(12, n * 0.5 + 1))
    ax.axis("off")

    table_data = [
        [i + 1, targets[i], predictions[i]]
        for i in range(n)
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=["#", "Ground Truth", "Prediction"],
        loc="center",
        cellLoc="left",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width([0, 1, 2])

    plt.title("Prediction Samples", fontsize=12, pad=10)

    output_path = output_dir / "prediction_samples.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path
