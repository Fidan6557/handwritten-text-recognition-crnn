import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR


def build_scheduler(optimizer, scheduler_type: str = "reduce_on_plateau", **kwargs):
    """
    Build a learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer instance.
        scheduler_type: 'reduce_on_plateau' or 'cosine'.

    Returns:
        A PyTorch scheduler.
    """

    if scheduler_type == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 3),
            min_lr=kwargs.get("min_lr", 1e-6),
        )

    if scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", 30),
            eta_min=kwargs.get("eta_min", 1e-6),
        )

    raise ValueError(f"Unknown scheduler type: {scheduler_type}")
