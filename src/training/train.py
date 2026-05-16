from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import IAMLineDataset, collate_fn
from src.models.crnn import CRNN
from src.training.evaluate import evaluate_model
from src.utils.charset import build_charset, save_charset
from src.utils.seed import set_seed


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
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_csv = "data/processed/train.csv"
    val_csv = "data/processed/val.csv"

    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    charset_path = Path("outputs/charset.json")
    charset_path.parent.mkdir(parents=True, exist_ok=True)

    char_to_idx, idx_to_char = build_charset([train_csv, val_csv])
    save_charset(char_to_idx, charset_path)

    train_dataset = IAMLineDataset(
        csv_path=train_csv,
        char_to_idx=char_to_idx,
    )

    val_dataset = IAMLineDataset(
        csv_path=val_csv,
        char_to_idx=char_to_idx,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = CRNN(
        num_classes=len(char_to_idx),
        hidden_size=256,
        num_lstm_layers=2,
    ).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-5,
    )

    best_cer = float("inf")
    epochs = 30

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

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"CER: {val_metrics['cer']:.4f} | "
            f"WER: {val_metrics['wer']:.4f}"
        )

        if val_metrics["cer"] < best_cer:
            best_cer = val_metrics["cer"]

            torch.save(
                model.state_dict(),
                checkpoint_dir / "best_crnn.pth",
            )

            print("Best model saved.")


if __name__ == "__main__":
    main()