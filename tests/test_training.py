import pytest
import torch
from stub_env import StubVecEnv

from tert.data import ContextWindow, Episode, ObservationNormalizer, TrajectoryWindowDataset
from tert.envs.obs_spec import A1
from tert.models import TeacherActorCritic, TerrainTransformer, TransformerConfig
from tert.training import (
    ImitationConfig,
    collect_online_correction,
    collect_teacher_rollouts,
    fit_to_teacher,
    masked_action_loss,
)

CONTEXT_LEN = 6


@pytest.fixture
def tert():
    return TerrainTransformer(
        TransformerConfig(obs_dim=A1.proprio_dim, context_len=CONTEXT_LEN, embed_dim=32, n_blocks=1)
    )


@pytest.fixture
def normalizer():
    norm = ObservationNormalizer(A1.proprio_dim)
    norm.update(torch.randn(256, A1.proprio_dim) * 3 + 1)
    return norm.freeze()


def test_normalizer_whitens():
    norm = ObservationNormalizer(4)
    data = torch.randn(10_000, 4) * torch.tensor([1.0, 5.0, 0.1, 2.0]) + 3.0
    for chunk in data.split(1000):  # streaming update must match a one-shot fit
        norm.update(chunk)
    out = norm(data)
    torch.testing.assert_close(out.mean(0), torch.zeros(4), atol=1e-4, rtol=0)
    torch.testing.assert_close(out.std(0), torch.ones(4), atol=1e-3, rtol=0)


def test_frozen_normalizer_refuses_refit(normalizer):
    with pytest.raises(RuntimeError, match="frozen"):
        normalizer.update(torch.randn(8, A1.proprio_dim))


def test_context_window_resets_per_env():
    ctx = ContextWindow(3, 4, 2, context_len=CONTEXT_LEN)
    for _ in range(CONTEXT_LEN):
        ctx.push_obs(torch.ones(3, 4))
        ctx.push_action(torch.ones(3, 2))

    ctx.reset(torch.tensor([1]))
    assert ctx.obs[1].abs().sum() == 0 and ctx.timestep[1] == 0
    assert ctx.obs[0].abs().sum() > 0 and ctx.timestep[0] == CONTEXT_LEN


def test_context_timesteps_clamp_during_warm_start():
    ctx = ContextWindow(1, 4, 2, context_len=CONTEXT_LEN)
    ctx.push_obs(torch.ones(1, 4))
    ctx.push_action(torch.ones(1, 2))
    assert ctx.timesteps().tolist() == [[0, 0, 0, 0, 0, 1]]


def make_episodes(lengths):
    return [
        Episode(
            torch.randn(n, A1.proprio_dim),
            torch.randn(n, A1.num_actions),
            torch.randn(n, A1.num_actions),
        )
        for n in lengths
    ]


def test_dataset_windows_are_front_padded(normalizer):
    short = 3
    ds = TrajectoryWindowDataset(make_episodes([short]), CONTEXT_LEN, normalizer)
    item = ds[0]

    assert item["obs"].shape == (CONTEXT_LEN, A1.proprio_dim)
    pad = CONTEXT_LEN - short
    assert item["mask"].tolist() == [0.0] * pad + [1.0] * short
    assert item["obs"][:pad].abs().sum() == 0  # padding at the front, not the back
    assert item["obs"][pad:].abs().sum() > 0
    assert item["timesteps"].tolist() == [0] * pad + list(range(short))


def test_dataset_full_windows_have_no_mask_holes(normalizer):
    ds = TrajectoryWindowDataset(make_episodes([50]), CONTEXT_LEN, normalizer, warm_start_prob=0.0)
    for _ in range(20):
        item = ds[0]
        assert item["mask"].sum() == CONTEXT_LEN
        assert torch.equal(item["timesteps"].diff(), torch.ones(CONTEXT_LEN - 1, dtype=torch.long))


def test_masked_loss_ignores_padding():
    target = torch.randn(2, CONTEXT_LEN, A1.num_actions)
    predicted = target.clone()
    mask = torch.ones(2, CONTEXT_LEN)
    mask[:, :2] = 0
    predicted[:, :2] = 99.0  # garbage where the mask is zero
    assert masked_action_loss(predicted, target, mask).item() == pytest.approx(0.0)


def test_stage1_collection_segments_episodes():
    env = StubVecEnv(num_envs=4, episode_length=5)
    episodes = collect_teacher_rollouts(env, TeacherActorCritic(), A1.proprio_dim, num_steps=40)

    assert episodes
    for ep in episodes:
        assert ep.obs.shape[1] == A1.proprio_dim
        assert len(ep) == ep.actions.shape[0] == ep.teacher_actions.shape[0]
        # stage 1 executes the teacher's own action, so the two must coincide
        torch.testing.assert_close(ep.actions, ep.teacher_actions)


def test_stage2_collection_diverges_from_teacher(tert, normalizer):
    env = StubVecEnv(num_envs=4, episode_length=5)
    episodes = collect_online_correction(
        env, tert, TeacherActorCritic(), normalizer, A1.proprio_dim, CONTEXT_LEN, num_steps=40
    )

    assert episodes
    # stage 2 executes TERT's action while labelling with the teacher's
    assert any(not torch.allclose(ep.actions, ep.teacher_actions) for ep in episodes)


def test_fit_to_teacher_reduces_loss(tert, normalizer):
    torch.manual_seed(0)
    ds = TrajectoryWindowDataset(make_episodes([30] * 8), CONTEXT_LEN, normalizer)
    cfg = ImitationConfig(batch_size=4, num_updates=60, warmup_steps=10, lr=1e-3)

    losses = []
    fit_to_teacher(tert, ds, cfg, on_update=lambda _, loss: losses.append(loss))

    assert len(losses) == cfg.num_updates
    assert sum(losses[-10:]) < sum(losses[:10])
