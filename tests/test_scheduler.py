import torch

from src.training.scheduler import build_scheduler


def test_reduce_on_plateau_scheduler_steps_without_verbose_error():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    scheduler = build_scheduler(optimizer, scheduler_type="reduce_on_plateau")

    scheduler.step(1.0)
