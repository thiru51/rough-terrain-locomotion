"""Uniform evaluation across policies with incompatible interfaces.

The policies being compared do not agree on what an "observation" is. The
transformer wants a rolling window of normalised proprioception; the teacher
wants the full privileged vector; the stacked baseline wants a flattened
history; the latent baselines want a history buffer *and* the current step. The
comparison is only fair if all of them meet the same environment through the
same loop, so each gets a thin adapter exposing `reset` and `act`.

Adapters own whatever buffers their policy needs, including clearing them per
environment on termination — without which a policy conditions across a reset
boundary and its numbers are quietly wrong.
"""

import time
from typing import Protocol

import torch

from tert.data.context import ContextWindow
from tert.eval.metrics import EvalMetrics, MetricAccumulator
from tert.models.baselines.stacked import ObservationStacker


class PolicyRunner(Protocol):
    def reset(self, env_ids: torch.Tensor | None = None) -> None: ...
    def act(self, obs: torch.Tensor) -> torch.Tensor: ...


class DirectRunner:
    """Stateless policies that consume a single observation: the teacher, flat PPO."""

    def __init__(self, policy):
        self.policy = policy

    def reset(self, env_ids=None):
        pass

    def act(self, obs):
        return self.policy.act_inference(obs)


class TransformerRunner:
    """The deployable policy: rolling observation-action window, normalised."""

    def __init__(
        self, model, normalizer, num_envs, proprio_dim, act_dim, context_len, device="cpu"
    ):
        self.model, self.normalizer, self.proprio_dim = model, normalizer, proprio_dim
        self.context = ContextWindow(num_envs, proprio_dim, act_dim, context_len, device)

    def reset(self, env_ids=None):
        self.context.reset(env_ids)

    def act(self, obs):
        return self.context.act(self.model, self.normalizer(obs[:, : self.proprio_dim]))


class StackedRunner:
    """Flattened fixed-length history; `history_length=1` is plain PPO."""

    def __init__(self, policy, num_envs, proprio_dim, act_dim, history_length, device="cpu"):
        self.policy, self.proprio_dim = policy, proprio_dim
        self.stacker = ObservationStacker(num_envs, proprio_dim, act_dim, history_length, device)
        self.last_action = torch.zeros(num_envs, act_dim, device=device)

    def reset(self, env_ids=None):
        self.stacker.reset(env_ids)
        if env_ids is None:
            self.last_action.zero_()
        elif env_ids.numel():
            self.last_action[env_ids] = 0.0

    def act(self, obs):
        stacked = self.stacker.push(obs[:, : self.proprio_dim], self.last_action)
        self.last_action = self.policy.act_inference(stacked)
        return self.last_action


class LatentRunner:
    """RMA-style: estimate the teacher's latent from a history buffer."""

    def __init__(self, policy, num_envs, proprio_dim, history_length, device="cpu"):
        self.policy, self.proprio_dim = policy, proprio_dim
        self.history = torch.zeros(num_envs, history_length, proprio_dim, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.history.zero_()
        elif env_ids.numel():
            self.history[env_ids] = 0.0

    def act(self, obs):
        self.history = self.history.roll(-1, dims=1)
        self.history[:, -1] = obs[:, : self.proprio_dim]
        return self.policy.act_inference(obs, history=self.history)


@torch.no_grad()
def evaluate(
    env, runner: PolicyRunner, num_steps: int, stop_when_all_done: bool = True
) -> EvalMetrics:
    """Roll `runner` through `env`, scoring each environment up to its first termination."""
    obs = env.reset()
    runner.reset()
    metrics = MetricAccumulator(env.num_envs, env.device)

    elapsed, forwards = 0.0, 0
    for _ in range(num_steps):
        start = time.perf_counter()
        actions = runner.act(obs)
        if obs.is_cuda:
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        forwards += 1

        obs, rewards, dones, infos = env.step(actions)
        metrics.step(
            actions,
            rewards,
            dones,
            timeouts=infos.get("time_outs"),
            torques=infos.get("torques"),
            dof_vel=infos.get("dof_vel"),
            base_lin_vel=infos.get("base_lin_vel"),
            commands=infos.get("commands"),
        )
        runner.reset(dones.nonzero(as_tuple=False).flatten())

        if stop_when_all_done and metrics.finished:
            break

    return metrics.result(inference_ms=1000.0 * elapsed / max(forwards, 1))
