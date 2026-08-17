import pytest
import torch

from tert.envs.obs_spec import A1
from tert.models import PrivilegedEncoder, TeacherActorCritic, TerrainTransformer, TransformerConfig


def test_obs_spec_matches_paper():
    assert (A1.proprio_dim, A1.privileged_dim, A1.total_dim) == (48, 203, 251)
    assert A1.heightmap_dim == 187


def make_batch(cfg, batch=4):
    t = torch.arange(cfg.context_len).expand(batch, -1)
    return (
        t,
        torch.randn(batch, cfg.context_len, cfg.obs_dim),
        torch.randn(batch, cfg.context_len, cfg.act_dim),
    )


@pytest.mark.parametrize("pre_ln", [False, True])
def test_transformer_shapes(pre_ln):
    cfg = TransformerConfig(pre_ln=pre_ln)
    model = TerrainTransformer(cfg)
    out = model(*make_batch(cfg))
    assert out.shape == (4, cfg.context_len, cfg.act_dim)


def test_action_head_is_causal():
    """a_hat_t must not depend on o_{>t} or a_{>=t}."""
    cfg = TransformerConfig()
    model = TerrainTransformer(cfg).eval()
    t, obs, act = make_batch(cfg)

    with torch.no_grad():
        base = model(t, obs, act)
        obs[:, 10:] = torch.randn_like(obs[:, 10:])
        act[:, 9:] = torch.randn_like(act[:, 9:])
        perturbed = model(t, obs, act)

    torch.testing.assert_close(base[:, :10], perturbed[:, :10])
    assert not torch.allclose(base[:, 10:], perturbed[:, 10:])


def test_attention_weights_are_causal_and_normalised():
    cfg = TransformerConfig()
    model = TerrainTransformer(cfg).eval()
    with torch.no_grad():
        _, weights = model(*make_batch(cfg), return_weights=True)

    assert len(weights) == cfg.n_blocks
    w = weights[0]
    assert w.shape == (4, cfg.n_heads, 2 * cfg.context_len, 2 * cfg.context_len)
    torch.testing.assert_close(w.sum(-1), torch.ones_like(w.sum(-1)))
    assert w.triu(diagonal=1).abs().max() == 0


def test_weight_return_path_matches_fast_path():
    cfg = TransformerConfig()
    model = TerrainTransformer(cfg).eval()
    batch = make_batch(cfg)
    with torch.no_grad():
        fast = model(*batch)
        explicit, _ = model(*batch, return_weights=True)
    torch.testing.assert_close(fast, explicit)


def test_encoder_reads_disjoint_modalities():
    enc = PrivilegedEncoder().eval()
    e = torch.zeros(2, A1.privileged_dim)
    with torch.no_grad():
        base = enc(e)
        e[:, A1.slices()["env_params"]] = 5.0  # only friction/mass/kp/kd perturbed
        assert not torch.allclose(base, enc(e))


def test_teacher_shapes_and_privileged_slicing():
    teacher = TeacherActorCritic()
    obs = torch.randn(8, A1.total_dim)
    assert teacher.act(obs).shape == (8, A1.num_actions)
    assert teacher.act_inference(obs).shape == (8, A1.num_actions)
    assert teacher.evaluate(obs).shape == (8,)
    assert teacher.action_log_prob(teacher.act(obs)).shape == (8,)
