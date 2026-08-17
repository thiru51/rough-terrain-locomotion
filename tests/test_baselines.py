import pytest
import torch

from tert.envs.obs_spec import A1
from tert.models import TeacherActorCritic, TerrainTransformer, TransformerConfig
from tert.models.baselines import (
    LatentPolicy,
    ObservationStacker,
    StackedActorCritic,
    TCNEncoder,
    TransformerLatentEstimator,
)

B, LATENT = 4, 12


def test_tcn_consumes_fifty_steps():
    tcn = TCNEncoder(obs_dim=A1.proprio_dim, history_length=50, latent_dim=LATENT)
    history = torch.randn(B, 50, A1.proprio_dim)
    assert tcn(history).shape == (B, LATENT)


@pytest.mark.parametrize("history_length", [40, 50, 64])
def test_tcn_flatten_width_is_derived_not_hard_coded(history_length):
    tcn = TCNEncoder(A1.proprio_dim, history_length, LATENT)
    assert tcn(torch.randn(2, history_length, A1.proprio_dim)).shape == (2, LATENT)


def test_tcn_rejects_a_history_it_cannot_convolve():
    """The strided stack collapses time fast; failing here beats an opaque conv error."""
    with pytest.raises(ValueError, match="too short"):
        TCNEncoder(A1.proprio_dim, history_length=30, latent_dim=LATENT)


def test_teacher_exposes_its_latent_as_a_regression_target():
    teacher = TeacherActorCritic()
    obs = torch.randn(B, A1.total_dim)
    assert teacher.latent(obs).shape == (B, LATENT)


def test_rma_policy_runs_through_the_frozen_teacher_actor():
    teacher = TeacherActorCritic()
    policy = LatentPolicy(TCNEncoder(A1.proprio_dim, 50, LATENT), teacher)
    obs = torch.randn(B, A1.total_dim)

    action = policy.act_inference(obs, history=torch.randn(B, 50, A1.proprio_dim))
    assert action.shape == (B, A1.num_actions)
    assert all(not p.requires_grad for p in policy.teacher.parameters())


def test_tert_latent_ablation_shares_the_rma_structure():
    """TERT-Latent differs from RMA only in the estimator."""
    teacher = TeacherActorCritic()
    context = 20
    transformer = TerrainTransformer(
        TransformerConfig(obs_dim=A1.proprio_dim, act_dim=LATENT, context_len=context)
    )
    policy = LatentPolicy(TransformerLatentEstimator(transformer), teacher)

    action = policy.act_inference(
        torch.randn(B, A1.total_dim),
        timesteps=torch.arange(context).expand(B, -1),
        obs=torch.randn(B, context, A1.proprio_dim),
        actions=torch.randn(B, context, LATENT),
    )
    assert action.shape == (B, A1.num_actions)


def test_latent_estimator_gradients_do_not_reach_the_teacher():
    teacher = TeacherActorCritic()
    policy = LatentPolicy(TCNEncoder(A1.proprio_dim, 50, LATENT), teacher)

    action = policy.act_inference(
        torch.randn(B, A1.total_dim), history=torch.randn(B, 50, A1.proprio_dim)
    )
    action.square().sum().backward()

    assert all(p.grad is None for p in policy.teacher.parameters())
    assert any(p.grad is not None for p in policy.estimator.parameters())


def test_stacker_flattens_obs_and_actions():
    stacker = ObservationStacker(B, A1.proprio_dim, A1.num_actions, history_length=3)
    stacked = stacker.push(torch.ones(B, A1.proprio_dim), torch.ones(B, A1.num_actions))

    assert stacked.shape == (B, stacker.stacked_dim)
    assert stacker.stacked_dim == 3 * (A1.proprio_dim + A1.num_actions)
    # Only the newest slot is populated after a single push.
    assert stacker.obs[:, -1].sum() > 0 and stacker.obs[:, 0].sum() == 0


def test_stacker_resets_per_env():
    stacker = ObservationStacker(3, 4, 2, history_length=2)
    stacker.push(torch.ones(3, 4), torch.ones(3, 2))
    stacker.reset(torch.tensor([1]))
    assert stacker.obs[1].abs().sum() == 0
    assert stacker.obs[0].abs().sum() > 0


@pytest.mark.parametrize("history_length", [1, 5])
def test_stacked_policy_is_ppo_compatible(history_length):
    """PPO is StackedPPO with history_length = 1; both need the same interface."""
    policy = StackedActorCritic(A1.proprio_dim, A1.num_actions, history_length)
    obs = torch.randn(B, history_length * (A1.proprio_dim + A1.num_actions))

    actions = policy.act(obs)
    assert actions.shape == (B, A1.num_actions)
    assert policy.evaluate(obs).shape == (B,)
    assert policy.action_log_prob(actions).shape == (B,)
    assert policy.entropy.shape == (B,)
    assert policy.action_mean.shape == policy.action_std.shape == (B, A1.num_actions)


def test_stacked_policy_sees_no_privileged_information():
    policy = StackedActorCritic(A1.proprio_dim, A1.num_actions, history_length=1)
    assert policy.actor[0].in_features == A1.proprio_dim + A1.num_actions
    assert policy.actor[0].in_features < A1.total_dim
