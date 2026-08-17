"""Rollout collection for both training passes.

The two stages differ in exactly one respect — who chooses the action:

    pass 1 (offline)            teacher acts, teacher labels
    pass 2 (online correction)  the transformer acts, teacher labels

The teacher labels every visited state either way, so both produce the same
`Episode` type and feed the same optimiser. That is the DAgger structure: pass 2
is not a different objective, it is the same objective evaluated on the states
the policy actually reaches.

Collection is tensor-first — steps are appended to a dense `(num_steps, num_envs, ·)`
buffer on device and segmented into episodes once, at the end. Segmenting per step
in Python costs `num_envs` small tensor ops per control step, which at 4096 envs
dominates the simulator. Budget roughly `num_steps · num_envs · 72 · 4` bytes.
"""

import torch

from tert.data.context import ContextWindow
from tert.data.dataset import Episode


class _Rollout:
    def __init__(self, num_steps, num_envs, obs_dim, act_dim, device):
        shape = (num_steps, num_envs)
        self.obs = torch.zeros(*shape, obs_dim, device=device)
        self.actions = torch.zeros(*shape, act_dim, device=device)
        self.teacher_actions = torch.zeros(*shape, act_dim, device=device)
        self.dones = torch.zeros(*shape, dtype=torch.bool, device=device)
        self.t = 0

    def add(self, obs, actions, teacher_actions, dones):
        i = self.t
        self.obs[i], self.actions[i] = obs, actions
        self.teacher_actions[i], self.dones[i] = teacher_actions, dones
        self.t += 1

    def episodes(self, min_len: int = 2) -> list[Episode]:
        """Split each environment's column at termination boundaries.

        The trailing fragment after the last `done` is discarded: it was cut by
        the collection budget rather than by the environment, and keeping it
        would bias the dataset toward truncated episodes.
        """
        obs, actions = self.obs[: self.t].cpu(), self.actions[: self.t].cpu()
        teacher, dones = self.teacher_actions[: self.t].cpu(), self.dones[: self.t].cpu()

        out = []
        for env in range(obs.shape[1]):
            ends = dones[:, env].nonzero(as_tuple=False).flatten().tolist()
            start = 0
            for end in ends:
                if end + 1 - start >= min_len:
                    s = slice(start, end + 1)
                    out.append(Episode(obs[s, env], actions[s, env], teacher[s, env]))
                start = end + 1
        return out


@torch.no_grad()
def collect_teacher_rollouts(env, teacher, proprio_dim: int, num_steps: int) -> list[Episode]:
    """Pass 1: the teacher drives, using privileged observations the policy never gets."""
    obs = env.reset()
    rollout = _Rollout(num_steps, env.num_envs, proprio_dim, env.num_actions, env.device)

    for _ in range(num_steps):
        action = teacher.act_inference(obs)
        next_obs, _, dones, _ = env.step(action)
        rollout.add(obs[:, :proprio_dim], action, action, dones)
        obs = next_obs

    return rollout.episodes()


@torch.no_grad()
def collect_online_correction(
    env, tert, teacher, normalizer, proprio_dim: int, context_len: int, num_steps: int
) -> list[Episode]:
    """Pass 2: the transformer drives; the teacher labels the states it actually visits.

    The teacher still reads privileged observations to produce its label, which is
    fine — labelling only ever happens in simulation. The transformer sees the
    normalised proprioceptive window and nothing else.
    """
    obs = env.reset()
    ctx = ContextWindow(env.num_envs, proprio_dim, env.num_actions, context_len, env.device)
    rollout = _Rollout(num_steps, env.num_envs, proprio_dim, env.num_actions, env.device)

    for _ in range(num_steps):
        proprio = obs[:, :proprio_dim]
        action = ctx.act(tert, normalizer(proprio))
        teacher_action = teacher.act_inference(obs)

        next_obs, _, dones, _ = env.step(action)
        rollout.add(proprio, action, teacher_action, dones)
        ctx.reset(dones.nonzero(as_tuple=False).flatten())  # history must not cross a reset
        obs = next_obs

    return rollout.episodes()
