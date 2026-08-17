"""Masked regression of the transformer onto the teacher's actions.

Both training passes minimise the same loss

    L = sum_t (a_hat_t - a_bar_t)^2

and differ only in the trajectory distribution the expectation is taken over, so
one optimiser serves both. Keeping it in one place makes the ablations
the single-pass ablations a matter of which passes are run, not of which code
path is taken.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class ImitationConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 10_000
    grad_clip: float = 0.25
    batch_size: int = 64
    num_updates: int = 200_000


def make_optimizer(model, cfg: ImitationConfig):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / cfg.warmup_steps, 1.0)
    )
    return optimizer, warmup


def masked_action_loss(predicted, target, mask):
    """MSE over valid timesteps only; padded context slots carry no target."""
    weights = mask.unsqueeze(-1).expand_as(target)
    return F.mse_loss(
        predicted * weights, target * weights, reduction="sum"
    ) / weights.sum().clamp_min(1)


def fit_to_teacher(model, dataset, cfg: ImitationConfig, device="cpu", on_update=None):
    """Run `cfg.num_updates` optimiser steps, cycling the dataset as needed."""
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    optimizer, scheduler = make_optimizer(model, cfg)
    model.train().to(device)

    batches = iter(loader)
    for update in range(cfg.num_updates):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)

        batch = {k: v.to(device) for k, v in batch.items()}
        predicted = model(batch["timesteps"], batch["obs"], batch["actions"])
        loss = masked_action_loss(predicted, batch["teacher_actions"], batch["mask"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if on_update is not None:
            on_update(update, loss.item())

    return model
