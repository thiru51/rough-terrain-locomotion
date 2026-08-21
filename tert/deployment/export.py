"""TorchScript export for the C++ inference node.

Normalisation is folded in as buffers, so the artifact the robot loads is one
self-contained file rather than a checkpoint plus a .npy that can go stale
against it.

torch.jit is deprecated in favour of torch.export, but LibTorch's
`torch::jit::load` still reads TorchScript, so this stays until the C++ side
moves to AOTInductor.
"""

from pathlib import Path

import torch
import torch.nn as nn


class DeploymentPolicy(nn.Module):
    """Inference-only wrapper: a window in, one action out.

    Timesteps are an input rather than computed here because the caller owns
    episode time. They are clamped to the embedding table: the index is absolute
    episode step, and a robot that runs longer than `max_timestep` would
    otherwise index out of bounds. Clamping saturates the positional signal
    rather than crashing, which is the better failure on hardware.
    """

    def __init__(self, model, normalizer, action_scale: float = 0.25):
        super().__init__()
        self.model = model
        self.action_scale = action_scale
        self.max_timestep = model.cfg.max_timestep
        self.register_buffer("obs_mean", normalizer.mean.clone())
        self.register_buffer("obs_std", normalizer.std.clone())

    def forward(self, timesteps, obs_window, action_window):
        obs = (obs_window - self.obs_mean) / self.obs_std
        timesteps = timesteps.clamp(0, self.max_timestep - 1)
        return self.model(timesteps, obs, action_window)[:, -1]


def export_policy(model, normalizer, path, context_len=None, obs_dim=None, act_dim=None):
    """Trace and save. Batch size and context length are baked in by tracing."""
    cfg = model.cfg
    context_len = context_len or cfg.context_len
    obs_dim = obs_dim or cfg.obs_dim
    act_dim = act_dim or cfg.act_dim

    wrapper = DeploymentPolicy(model, normalizer).eval()
    example = (
        torch.arange(context_len).unsqueeze(0),
        torch.zeros(1, context_len, obs_dim),
        torch.zeros(1, context_len, act_dim),
    )
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(path))
    return path


def check_export(traced_path, model, normalizer, tolerance=1e-5, seed=0):
    """Compare the traced artifact against eager output on random input.

    Tracing silently drops control flow, so this is not optional.
    """
    torch.manual_seed(seed)
    cfg = model.cfg
    timesteps = torch.arange(cfg.context_len).unsqueeze(0)
    obs = torch.randn(1, cfg.context_len, cfg.obs_dim)
    actions = torch.randn(1, cfg.context_len, cfg.act_dim)

    eager = DeploymentPolicy(model, normalizer).eval()
    loaded = torch.jit.load(str(traced_path))
    with torch.no_grad():
        difference = (eager(timesteps, obs, actions) - loaded(timesteps, obs, actions)).abs().max()
    return float(difference) < tolerance, float(difference)
