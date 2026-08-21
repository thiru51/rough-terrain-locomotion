import pytest
import torch

from tert.data import ContextWindow, ObservationNormalizer
from tert.deployment.export import DeploymentPolicy, check_export, export_policy
from tert.envs.obs_spec import A1
from tert.models import TerrainTransformer, TransformerConfig

CONTEXT = 8


@pytest.fixture
def model():
    return TerrainTransformer(
        TransformerConfig(obs_dim=A1.proprio_dim, context_len=CONTEXT, embed_dim=32, n_blocks=2)
    ).eval()


@pytest.fixture
def normalizer():
    norm = ObservationNormalizer(A1.proprio_dim)
    norm.update(torch.randn(256, A1.proprio_dim) * 2 + 1)
    return norm.freeze()


def windows(batch=1):
    return (
        torch.arange(CONTEXT).expand(batch, -1),
        torch.randn(batch, CONTEXT, A1.proprio_dim),
        torch.randn(batch, CONTEXT, A1.num_actions),
    )


def test_wrapper_returns_one_action(model, normalizer):
    policy = DeploymentPolicy(model, normalizer).eval()
    with torch.no_grad():
        assert policy(*windows()).shape == (1, A1.num_actions)


def test_normalisation_is_baked_in(model, normalizer):
    """The exported artifact must not need a companion .npy that can go stale."""
    policy = DeploymentPolicy(model, normalizer).eval()
    timesteps, obs, actions = windows()

    with torch.no_grad():
        folded = policy(timesteps, obs, actions)
        manual = model(timesteps, normalizer(obs), actions)[:, -1]
    torch.testing.assert_close(folded, manual)


def test_timesteps_clamp_beyond_the_embedding_table(model, normalizer):
    """A robot outruns max_timestep; saturating beats indexing out of bounds."""
    policy = DeploymentPolicy(model, normalizer).eval()
    _, obs, actions = windows()
    huge = torch.full((1, CONTEXT), model.cfg.max_timestep + 5_000)

    with torch.no_grad():
        action = policy(huge, obs, actions)
    assert action.shape == (1, A1.num_actions)
    assert torch.isfinite(action).all()


def test_traced_module_matches_eager(model, normalizer, tmp_path):
    path = export_policy(model, normalizer, tmp_path / "policy.pt")
    assert path.exists()

    matches, difference = check_export(path, model, normalizer)
    assert matches, f"traced output diverged by {difference}"


def test_exported_module_loads_without_the_source_class(model, normalizer, tmp_path):
    """TorchScript must be self-contained — the C++ node has no Python."""
    path = export_policy(model, normalizer, tmp_path / "policy.pt")
    loaded = torch.jit.load(str(path))

    with torch.no_grad():
        assert loaded(*windows()).shape == (1, A1.num_actions)


def test_export_agrees_with_the_python_context_window(model, normalizer, tmp_path):
    """The C++ buffer mirrors ContextWindow; both must feed the model identically."""
    path = export_policy(model, normalizer, tmp_path / "policy.pt")
    loaded = torch.jit.load(str(path))

    context = ContextWindow(1, A1.proprio_dim, A1.num_actions, CONTEXT)
    for _ in range(3):
        # Step the window by hand: act() commits the action and advances time,
        # so the window has to be read between the push and the commit.
        context.push_obs(normalizer(torch.randn(1, A1.proprio_dim)))
        timesteps, obs, actions = context.timesteps(), context.obs, context.actions

        with torch.no_grad():
            expected = model(timesteps, obs, actions)[:, -1]
            raw = obs * normalizer.std + normalizer.mean  # the artifact normalises internally
            actual = loaded(timesteps, raw, actions)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        context.push_action(expected)
