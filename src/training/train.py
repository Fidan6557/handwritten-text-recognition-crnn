import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import IAMLineDataset, collate_fn
from src.models.crnn import CRNN
from src.training.evaluate import evaluate_model
from src.training.scheduler import build_scheduler
from src.utils.charset import build_charset, save_charset
from src.utils.logger import TrainingLogger
from src.utils.seed import set_seed


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train OCR model for one epoch.
    """

    model.train()
    total_loss = 0.0

    for batch in dataloader:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"].to(device)

        log_probs = model(images)

        time_steps = log_probs.size(0)
        batch_size = log_probs.size(1)

        input_lengths = torch.full(
            size=(batch_size,),
            fill_value=time_steps,
            dtype=torch.long,
            device=device,
        )

        loss = criterion(
            log_probs,
            labels,
            input_lengths,
            label_lengths,
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train CRNN model for handwritten text recognition.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config file (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--scheduler",
        default="reduce_on_plateau",
        choices=["reduce_on_plateau", "cosine"],
        help="Learning rate scheduler type (default: reduce_on_plateau).",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    set_seed(config["project"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_csv = str(Path(config["data"]["processed_dir"]) / "train.csv")
    val_csv = str(Path(config["data"]["processed_dir"]) / "val.csv")

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    charset_path = Path(config["paths"]["charset"])
    charset_path.parent.mkdir(parents=True, exist_ok=True)

    char_to_idx, idx_to_char = build_charset([train_csv, val_csv])
    save_charset(char_to_idx, charset_path)

    train_dataset = IAMLineDataset(
        csv_path=train_csv,
        char_to_idx=char_to_idx,
        image_width=config["data"]["image_width"],
        image_height=config["data"]["image_height"],
    )

    val_dataset = IAMLineDataset(
        csv_path=val_csv,
        char_to_idx=char_to_idx,
        image_width=config["data"]["image_width"],
        image_height=config["data"]["image_height"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = CRNN(
        num_classes=len(char_to_idx),
        hidden_size=config["model"]["hidden_size"],
        num_lstm_layers=config["model"]["num_lstm_layers"],
    ).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = build_scheduler(
        optimizer,
        scheduler_type=args.scheduler,
        T_max=config["training"]["epochs"],
    )

    logger = TrainingLogger(log_dir="outputs/logs")

    best_cer = float("inf")
    epochs = config["training"]["epochs"]

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            idx_to_char=idx_to_char,
            device=device,
        )

        metrics = {
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "cer": round(val_metrics["cer"], 6),
            "wer": round(val_metrics["wer"], 6),
        }

        logger.log(epoch=epoch, metrics=metrics)

        if args.scheduler == "reduce_on_plateau":
            scheduler.step(val_metrics["cer"])
        else:
            scheduler.step()

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {metrics['train_loss']:.4f} | "
            f"Val Loss: {metrics['val_loss']:.4f} | "
            f"CER: {metrics['cer']:.4f} | "
            f"WER: {metrics['wer']:.4f}"
        )

        if val_metrics["cer"] < best_cer:
            best_cer = val_metrics["cer"]
            torch.save(model.state_dict(), checkpoint_dir / "best_crnn.pth")
            print(f"  → Best model saved (CER: {best_cer:.4f})")

    print(f"\nTraining complete. Log saved to: {logger.get_log_path()}")


if __name__ == "__main__":
    main()