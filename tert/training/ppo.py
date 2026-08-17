"""PPO for the teacher policy (paper Sec. IV-A).

Follows the rsl_rl configuration the official release trains with: clipped
surrogate objective, GAE, clipped value loss, and a learning rate driven by
measured KL divergence rather than a fixed schedule.

Only the teacher is trained by RL. TERT itself never sees a policy gradient —
it is fit by regression onto the teacher's actions (`tert/training/imitation.py`).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    clip_param: float = 0.2
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1e-3
    schedule: str = "adaptive"  # "adaptive" | "fixed"
    desired_kl: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    lr_range: tuple[float, float] = (1e-5, 1e-2)


class RolloutStorage:
    """One PPO iteration of on-policy transitions, shaped (num_steps, num_envs, ·)."""

    def __init__(self, num_steps, num_envs, obs_dim, act_dim, gamma=0.99, device="cpu"):
        self.num_steps, self.num_envs, self.gamma = num_steps, num_envs, gamma

        def zeros(*trailing):
            return torch.zeros(num_steps, num_envs, *trailing, device=device)

        self.obs, self.actions = zeros(obs_dim), zeros(act_dim)
        self.rewards, self.dones = zeros(), zeros()
        self.values, self.log_probs = zeros(), zeros()
        self.mu, self.sigma = zeros(act_dim), zeros(act_dim)
        self.returns, self.advantages = zeros(), zeros()
        self.step = 0

    def add(self, obs, actions, rewards, dones, values, log_probs, mu, sigma, timeouts=None):
        i = self.step
        if timeouts is not None:
            # A truncated episode has not actually ended, so bootstrap its value
            # back in. Without this the agent learns that the time limit is a
            # terminal state worth avoiding.
            rewards = rewards + self.gamma * values * timeouts.float()

        self.obs[i], self.actions[i] = obs, actions
        self.rewards[i], self.dones[i] = rewards, dones.float()
        self.values[i], self.log_probs[i] = values, log_probs
        self.mu[i], self.sigma[i] = mu, sigma
        self.step += 1

    def compute_returns(self, last_values, gamma, lam):
        """Generalised advantage estimation, walked backwards through the rollout."""
        advantage = torch.zeros_like(last_values)
        for t in reversed(range(self.num_steps)):
            next_values = last_values if t == self.num_steps - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * not_done - self.values[t]
            advantage = delta + gamma * lam * not_done * advantage
            self.returns[t] = advantage + self.values[t]

        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def mini_batches(self, num_mini_batches, num_epochs):
        """Flatten across envs and steps, then yield shuffled minibatches per epoch."""
        total = self.num_steps * self.num_envs
        size = total // num_mini_batches
        flat = {
            name: getattr(self, name).flatten(0, 1)
            for name in (
                "obs",
                "actions",
                "log_probs",
                "values",
                "returns",
                "advantages",
                "mu",
                "sigma",
            )
        }
        for _ in range(num_epochs):
            order = torch.randperm(total, device=self.obs.device)
            for start in range(0, size * num_mini_batches, size):
                idx = order[start : start + size]
                yield {name: value[idx] for name, value in flat.items()}


class PPO:
    def __init__(self, policy: nn.Module, cfg: PPOConfig, device="cpu"):
        self.policy = policy.to(device)
        self.cfg = cfg
        self.learning_rate = cfg.learning_rate
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    def _adapt_learning_rate(self, batch, mu, sigma):
        """Move the step size so the policy update stays near `desired_kl`.

        A fixed learning rate either stalls or destabilises as the action
        distribution's scale drifts over training; rsl_rl measures the actual KL
        between the old and new Gaussians and rescales accordingly.
        """
        if self.cfg.schedule != "adaptive":
            return

        with torch.no_grad():
            kl = torch.sum(
                torch.log(sigma / batch["sigma"] + 1e-5)
                + (batch["sigma"].square() + (batch["mu"] - mu).square()) / (2.0 * sigma.square())
                - 0.5,
                dim=-1,
            ).mean()

        lo, hi = self.cfg.lr_range
        if kl > self.cfg.desired_kl * 2.0:
            self.learning_rate = max(lo, self.learning_rate / 1.5)
        elif 0.0 < kl < self.cfg.desired_kl / 2.0:
            self.learning_rate = min(hi, self.learning_rate * 1.5)

        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def _value_loss(self, batch, values):
        if not self.cfg.use_clipped_value_loss:
            return (batch["returns"] - values).square().mean()
        clipped = batch["values"] + (values - batch["values"]).clamp(
            -self.cfg.clip_param, self.cfg.clip_param
        )
        return torch.max(
            (values - batch["returns"]).square(), (clipped - batch["returns"]).square()
        ).mean()

    def update(self, storage: RolloutStorage) -> dict[str, float]:
        totals = {"surrogate": 0.0, "value": 0.0, "entropy": 0.0}
        count = 0

        for batch in storage.mini_batches(self.cfg.num_mini_batches, self.cfg.num_learning_epochs):
            self.policy.act(batch["obs"])  # repopulates the distribution
            log_probs = self.policy.action_log_prob(batch["actions"])
            values = self.policy.evaluate(batch["obs"])
            mu, sigma = self.policy.action_mean, self.policy.action_std
            entropy = self.policy.entropy.mean()

            self._adapt_learning_rate(batch, mu, sigma)

            ratio = torch.exp(log_probs - batch["log_probs"])
            clipped_ratio = ratio.clamp(1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param)
            surrogate = -torch.min(
                batch["advantages"] * ratio, batch["advantages"] * clipped_ratio
            ).mean()
            value_loss = self._value_loss(batch, values)

            loss = (
                surrogate + self.cfg.value_loss_coef * value_loss - self.cfg.entropy_coef * entropy
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()

            totals["surrogate"] += surrogate.item()
            totals["value"] += value_loss.item()
            totals["entropy"] += entropy.item()
            count += 1

        return {**{k: v / count for k, v in totals.items()}, "learning_rate": self.learning_rate}
