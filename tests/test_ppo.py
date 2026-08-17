import pytest
import torch
from stub_env import StubVecEnv

from tert.envs.obs_spec import A1
from tert.models import TeacherActorCritic
from tert.training.ppo import PPO, PPOConfig, RolloutStorage
from tert.training.teacher_runner import TeacherTrainConfig, train_teacher

STEPS, ENVS, ACT = 8, 4, A1.num_actions


def make_storage(gamma=0.99, **fill):
    storage = RolloutStorage(STEPS, ENVS, A1.total_dim, ACT, gamma=gamma)
    for _ in range(STEPS):
        storage.add(
            obs=torch.randn(ENVS, A1.total_dim),
            actions=torch.randn(ENVS, ACT),
            rewards=fill.get("rewards", torch.ones(ENVS)),
            dones=fill.get("dones", torch.zeros(ENVS, dtype=torch.bool)),
            values=fill.get("values", torch.zeros(ENVS)),
            log_probs=torch.randn(ENVS),
            mu=torch.zeros(ENVS, ACT),
            sigma=torch.ones(ENVS, ACT),
            timeouts=fill.get("timeouts"),
        )
    return storage


def test_gae_discounts_a_constant_reward_stream():
    """With zero values and no termination, returns are a geometric sum of rewards."""
    gamma, lam = 0.9, 1.0
    storage = make_storage(gamma=gamma)
    storage.compute_returns(torch.zeros(ENVS), gamma, lam)

    remaining = STEPS - 1  # index of the last step
    expected = sum(gamma**k for k in range(remaining + 1))
    assert storage.returns[0, 0].item() == pytest.approx(expected, rel=1e-5)
    # Later steps have fewer rewards left to collect.
    assert (storage.returns[0] > storage.returns[-1]).all()


def test_termination_cuts_the_return_bootstrap():
    gamma = 0.99
    ongoing = make_storage(gamma=gamma)
    ongoing.compute_returns(torch.zeros(ENVS), gamma, 0.95)

    terminating = make_storage(gamma=gamma, dones=torch.ones(ENVS, dtype=torch.bool))
    terminating.compute_returns(torch.zeros(ENVS), gamma, 0.95)

    # Every step ends the episode, so no future reward may be credited.
    assert terminating.returns.allclose(torch.ones(STEPS, ENVS))
    assert (ongoing.returns[0] > terminating.returns[0]).all()


def test_timeout_bootstraps_value_back_into_reward():
    """A truncated episode is not a failure; its value estimate must survive."""
    values = torch.full((ENVS,), 5.0)
    plain = make_storage(values=values)
    truncated = make_storage(values=values, timeouts=torch.ones(ENVS, dtype=torch.bool))
    assert (truncated.rewards[0] - plain.rewards[0]).allclose(torch.full((ENVS,), 0.99 * 5.0))


def test_advantages_are_normalised():
    storage = make_storage(rewards=torch.randn(ENVS))
    storage.compute_returns(torch.randn(ENVS), 0.99, 0.95)
    assert storage.advantages.mean().abs() < 1e-5
    assert storage.advantages.std() == pytest.approx(1.0, rel=1e-3)


def test_minibatches_partition_the_rollout():
    storage = make_storage()
    storage.compute_returns(torch.zeros(ENVS), 0.99, 0.95)
    batches = list(storage.mini_batches(num_mini_batches=4, num_epochs=2))

    assert len(batches) == 8
    assert batches[0]["obs"].shape == (STEPS * ENVS // 4, A1.total_dim)
    # One epoch must cover every transition exactly once.
    seen = torch.cat([b["log_probs"] for b in batches[:4]]).sort().values
    expected = storage.log_probs.flatten().sort().values
    torch.testing.assert_close(seen, expected)


def test_adaptive_lr_shrinks_on_large_kl():
    policy = TeacherActorCritic()
    ppo = PPO(policy, PPOConfig(learning_rate=1e-3))
    batch = {"mu": torch.zeros(4, ACT), "sigma": torch.ones(4, ACT)}

    # New distribution far from the old one -> KL above target -> lower lr.
    ppo._adapt_learning_rate(batch, mu=torch.full((4, ACT), 5.0), sigma=torch.ones(4, ACT))
    assert ppo.learning_rate < 1e-3
    assert ppo.optimizer.param_groups[0]["lr"] == ppo.learning_rate


def test_adaptive_lr_grows_when_updates_are_timid():
    policy = TeacherActorCritic()
    ppo = PPO(policy, PPOConfig(learning_rate=1e-4))
    batch = {"mu": torch.zeros(4, ACT), "sigma": torch.ones(4, ACT)}

    ppo._adapt_learning_rate(batch, mu=torch.zeros(4, ACT), sigma=torch.ones(4, ACT) * 1.0001)
    assert ppo.learning_rate > 1e-4


def test_adaptive_lr_respects_bounds():
    ppo = PPO(TeacherActorCritic(), PPOConfig(learning_rate=1e-5))
    batch = {"mu": torch.zeros(4, ACT), "sigma": torch.ones(4, ACT)}
    for _ in range(20):
        ppo._adapt_learning_rate(batch, torch.full((4, ACT), 9.0), torch.ones(4, ACT))
    assert ppo.learning_rate >= ppo.cfg.lr_range[0]


def test_fixed_schedule_leaves_lr_alone():
    ppo = PPO(TeacherActorCritic(), PPOConfig(schedule="fixed", learning_rate=1e-3))
    batch = {"mu": torch.zeros(4, ACT), "sigma": torch.ones(4, ACT)}
    ppo._adapt_learning_rate(batch, torch.full((4, ACT), 9.0), torch.ones(4, ACT))
    assert ppo.learning_rate == 1e-3


def test_ppo_update_runs_and_changes_parameters():
    torch.manual_seed(0)
    policy = TeacherActorCritic()
    before = policy.actor[0].weight.detach().clone()

    storage = make_storage(rewards=torch.randn(ENVS))
    storage.compute_returns(torch.randn(ENVS), 0.99, 0.95)
    stats = PPO(policy, PPOConfig()).update(storage)

    assert set(stats) == {"surrogate", "value", "entropy", "learning_rate"}
    assert not torch.allclose(before, policy.actor[0].weight)


def test_teacher_training_loop_end_to_end():
    torch.manual_seed(0)
    env = StubVecEnv(num_envs=4, episode_length=5)
    cfg = TeacherTrainConfig(num_steps_per_env=6, max_iterations=3)

    seen = []
    train_teacher(env, TeacherActorCritic(), cfg, on_iteration=lambda i, s: seen.append(s))

    assert len(seen) == 3
    assert all(torch.isfinite(torch.tensor(s["surrogate"])) for s in seen)
