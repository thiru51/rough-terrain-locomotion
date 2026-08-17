"""Minimal VecEnv satisfying the backend contract, for testing without a simulator.

Dynamics are meaningless; only shapes, dtypes, termination timing and the `infos`
contract are real.
"""

import torch

from tert.envs.obs_spec import A1


class StubVecEnv:
    def __init__(self, num_envs=4, episode_length=12, spec=A1, device="cpu", timeout_at=None):
        self.num_envs = num_envs
        self.num_obs = spec.total_dim
        self.num_actions = spec.num_actions
        self.device = torch.device(device)
        self.dt = 0.02
        self.max_episode_length = episode_length
        self.episode_length = episode_length
        self.timeout_at = timeout_at  # if set, terminations at this step are timeouts
        self._step = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def reset(self):
        self._step.zero_()
        return torch.randn(self.num_envs, self.num_obs, device=self.device)

    def step(self, actions):
        assert actions.shape == (self.num_envs, self.num_actions)
        self._step += 1
        # Stagger terminations so episode segmentation is exercised on ragged data.
        dones = (self._step % (self.episode_length + torch.arange(self.num_envs))) == 0
        timeouts = (
            dones & (self._step == self.timeout_at)
            if self.timeout_at is not None
            else torch.zeros_like(dones)
        )
        self._step[dones] = 0

        obs = torch.randn(self.num_envs, self.num_obs, device=self.device)
        rewards = torch.ones(self.num_envs, device=self.device)
        infos = {
            "time_outs": timeouts,
            "torques": torch.ones(self.num_envs, self.num_actions, device=self.device),
            "dof_vel": torch.ones(self.num_envs, self.num_actions, device=self.device),
            "base_lin_vel": torch.zeros(self.num_envs, 3, device=self.device),
            "commands": torch.tensor([[0.4, 0.0, 0.0]], device=self.device).repeat(
                self.num_envs, 1
            ),
        }
        return obs, rewards, dones, infos
