from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


def build_scheduler(optimizer, scheduler_type="reduce_on_plateau", T_max=30):
    if scheduler_type == "reduce_on_plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type!r}. Choose 'reduce_on_plateau' or 'cosine'.")
