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


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
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

        loss = criterion(log_probs, labels, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the CRNN handwriting OCR model.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML training configuration.",
    )
    return parser.parse_args()


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)

    set_seed(cfg["project"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processed_dir = Path(cfg["data"]["processed_dir"])
    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"

    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = Path(cfg["paths"]["best_model"])
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    charset_path = Path(cfg["paths"]["charset"])
    charset_path.parent.mkdir(parents=True, exist_ok=True)

    char_to_idx, idx_to_char = build_charset([train_csv, val_csv])
    save_charset(char_to_idx, charset_path)

    batch_size = cfg["training"]["batch_size"]

    image_width = cfg["data"]["image_width"]
    image_height = cfg["data"]["image_height"]

    train_dataset = IAMLineDataset(
        csv_path=train_csv,
        char_to_idx=char_to_idx,
        image_width=image_width,
        image_height=image_height,
    )
    val_dataset = IAMLineDataset(
        csv_path=val_csv,
        char_to_idx=char_to_idx,
        image_width=image_width,
        image_height=image_height,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = CRNN(
        num_classes=len(char_to_idx),
        hidden_size=cfg["model"]["hidden_size"],
        num_lstm_layers=cfg["model"]["num_lstm_layers"],
        dropout=cfg["model"].get("dropout", 0.3),
    ).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    best_cer = float("inf")
    epochs = cfg["training"]["epochs"]
    scheduler_type = cfg["training"].get("scheduler", "reduce_on_plateau")
    scheduler = build_scheduler(optimizer, scheduler_type=scheduler_type, T_max=epochs)
    logger = TrainingLogger(log_dir=cfg["paths"]["logs_dir"])

    print(f"Device: {device} | Epochs: {epochs} | Batch: {batch_size}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_metrics = evaluate_model(model, val_loader, criterion, idx_to_char, device)

        if scheduler_type == "reduce_on_plateau":
            scheduler.step(val_metrics["cer"])
        else:
            scheduler.step()

        logger.log(epoch, {
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_metrics["loss"], 6),
            "cer":        round(val_metrics["cer"], 6),
            "wer":        round(val_metrics["wer"], 6),
        })

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"CER: {val_metrics['cer']:.4f} | "
            f"WER: {val_metrics['wer']:.4f}"
        )

        if val_metrics["cer"] < best_cer:
            best_cer = val_metrics["cer"]
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Best model saved (CER: {best_cer:.4f})")

    print(f"\nTraining complete. Best CER: {best_cer:.4f}")
    print(f"Log: {logger.get_log_path()}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
