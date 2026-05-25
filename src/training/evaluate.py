import torch

from src.inference.decoder import greedy_ctc_decode
from src.utils.metrics import character_error_rate, word_error_rate


def evaluate_model(model, dataloader, criterion, idx_to_char, device):
    """
    Evaluate OCR model using validation loss, CER and WER.
    """

    model.eval()

    total_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)
            targets = batch["texts"]

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

            predictions = greedy_ctc_decode(log_probs.cpu(), idx_to_char)

            for prediction, target in zip(predictions, targets):
                total_cer += character_error_rate(prediction, target)
                total_wer += word_error_rate(prediction, target)
                total_samples += 1

            total_loss += loss.item()

    return {
        "loss": total_loss / len(dataloader),
        "cer": total_cer / total_samples,
        "wer": total_wer / total_samples,
    }