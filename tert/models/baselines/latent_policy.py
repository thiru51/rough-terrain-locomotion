"""Policies that estimate the privileged latent instead of the action.

Both baselines here share one structure and differ only in the sequence model
that produces `l_hat`:

    TCN estimator          convolution over the last 50 observations
    Transformer estimator   attention over the context window

Both feed `proprio | l_hat` to the *teacher's own actor*, kept frozen. Keeping
them around separates two questions that are otherwise confounded: does the
attention help, or does predicting the action end to end rather than a latent
help? Swapping one factor at a time answers each.

It also makes the weakness of this design visible. The actor sits downstream of
`l_hat` and is frozen, so there is structurally no way to recover from a bad
estimate — which is why the main policy skips the latent entirely.
"""

import torch
import torch.nn as nn


class TransformerLatentEstimator(nn.Module):
    """Adapts a causal Transformer to emit a latent at the last context position."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, timesteps, obs, actions):
        return self.transformer(timesteps, obs, actions)[:, -1]


class LatentPolicy(nn.Module):
    def __init__(self, estimator: nn.Module, teacher, freeze_teacher: bool = True):
        super().__init__()
        self.estimator = estimator
        self.teacher = teacher
        self.spec = teacher.spec
        if freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            self.teacher.eval()

    def latent(self, **history):
        return self.estimator(**history)

    def act_inference(self, current_obs, **history):
        """`current_obs` is this step; `history` is forwarded verbatim to the estimator.

        The parameter is not called `obs` because the Transformer estimator takes
        a keyword of that name, and the two would collide.
        """
        proprio = current_obs[:, : self.spec.proprio_dim]
        features = torch.cat([proprio, self.estimator(**history)], dim=-1)
        return self.teacher.actor(features)


def latent_regression_loss(estimated, target):
    """RMA's objective: match the teacher's latent, not its action."""
    return nn.functional.mse_loss(estimated, target)
