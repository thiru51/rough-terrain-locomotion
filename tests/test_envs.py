import numpy as np
import pytest
import torch

from tert.envs.rewards import TERMS, RewardComposer, RobotState
from tert.envs.terrain import TerrainConfig, TerrainGrid, make_sub_terrain, select_type

DT = 0.02
N = 4


def make_state(**overrides) -> RobotState:
    zeros3 = torch.zeros(N, 3)
    zeros12 = torch.zeros(N, 12)
    zeros4 = torch.zeros(N, 4)
    defaults = dict(
        base_lin_vel=zeros3.clone(),
        base_ang_vel=zeros3.clone(),
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]).repeat(N, 1),
        base_height=torch.full((N,), 0.25),
        commands=zeros3.clone(),
        dof_pos=zeros12.clone(),
        dof_vel=zeros12.clone(),
        last_dof_vel=zeros12.clone(),
        torques=zeros12.clone(),
        last_torques=zeros12.clone(),
        actions=zeros12.clone(),
        last_actions=zeros12.clone(),
        feet_air_time=zeros4.clone(),
        first_contact=zeros4.clone(),
        contact_forces=torch.zeros(N, 17, 3),
        penalised_contact=torch.zeros(N, 8, 3),
        reset=torch.zeros(N, dtype=torch.bool),
        timeout=torch.zeros(N, dtype=torch.bool),
        dt=DT,
    )
    return RobotState(**{**defaults, **overrides})


# --- rewards ----------------------------------------------------------------


def test_velocity_tracking_peaks_at_zero_error():
    commands = torch.tensor([[0.4, 0.0, 0.0]]).repeat(N, 1)
    perfect = make_state(commands=commands, base_lin_vel=commands.clone())
    lagging = make_state(commands=commands, base_lin_vel=torch.zeros(N, 3))

    assert TERMS["tracking_lin_vel"](perfect).allclose(torch.ones(N))
    assert (TERMS["tracking_lin_vel"](lagging) < 1.0).all()


def test_timeout_is_not_a_failure():
    """A truncated episode must not be scored like a fall."""
    fell = make_state(reset=torch.ones(N, dtype=torch.bool))
    timed_out = make_state(
        reset=torch.ones(N, dtype=torch.bool), timeout=torch.ones(N, dtype=torch.bool)
    )
    assert TERMS["termination"](fell).sum() == N
    assert TERMS["termination"](timed_out).sum() == 0


def test_air_time_requires_a_command():
    air = torch.full((N, 4), 0.9)
    contact = torch.ones(N, 4)
    standing = make_state(feet_air_time=air, first_contact=contact)
    walking = make_state(
        feet_air_time=air,
        first_contact=contact,
        commands=torch.tensor([[0.4, 0.0, 0.0]]).repeat(N, 1),
    )
    assert TERMS["feet_air_time"](standing).sum() == 0
    assert (TERMS["feet_air_time"](walking) > 0).all()


def test_tert_smoothness_terms_fire_on_chatter():
    steady = make_state(actions=torch.zeros(N, 12), torques=torch.zeros(N, 12))
    chattering = make_state(
        actions=torch.ones(N, 12), torques=torch.ones(N, 12), last_torques=-torch.ones(N, 12)
    )
    assert TERMS["action_magnitude"](steady).sum() == 0
    assert TERMS["torques_smooth"](steady).sum() == 0
    assert (TERMS["action_magnitude"](chattering) > 0).all()
    assert (TERMS["torques_smooth"](chattering) > 0).all()


def test_composer_scales_by_dt():
    composer = RewardComposer({"lin_vel_z": -2.0}, dt=DT, only_positive=False)
    state = make_state(base_lin_vel=torch.tensor([[0.0, 0.0, 1.0]]).repeat(N, 1))
    total, parts = composer(state)
    assert total.allclose(torch.full((N,), -2.0 * DT))
    assert set(parts) == {"lin_vel_z"}


def test_composer_clips_negative_return():
    composer = RewardComposer({"lin_vel_z": -2.0}, dt=DT, only_positive=True)
    state = make_state(base_lin_vel=torch.tensor([[0.0, 0.0, 5.0]]).repeat(N, 1))
    assert composer(state)[0].allclose(torch.zeros(N))


def test_composer_rejects_unknown_terms():
    with pytest.raises(KeyError, match="typo_term"):
        RewardComposer({"typo_term": 1.0}, dt=DT)


def test_composer_skips_zero_weighted_terms():
    composer = RewardComposer({"lin_vel_z": -2.0, "orientation": 0.0}, dt=DT)
    assert "orientation" not in composer(make_state())[1]


# --- terrain ----------------------------------------------------------------


@pytest.fixture
def cfg():
    return TerrainConfig(num_rows=4, num_cols=5, terrain_length=4.0, border_size=1.0)


def test_type_selection_follows_proportions(cfg):
    types = [select_type(c, cfg) for c in range(cfg.num_cols)]
    assert types[0] == "smooth_slope"
    assert set(types) <= set(
        ["smooth_slope", "rough_slope", "stairs_up", "stairs_down", "discrete_obstacles"]
    )


@pytest.mark.parametrize(
    "terrain_type", ["smooth_slope", "rough_slope", "stairs_up", "stairs_down"]
)
def test_difficulty_increases_relief(terrain_type, cfg):
    rng = np.random.default_rng(0)
    easy = make_sub_terrain(terrain_type, 0.0, cfg, rng)
    hard = make_sub_terrain(terrain_type, 0.9, cfg, np.random.default_rng(0))
    assert np.ptp(hard) > np.ptp(easy)


def test_spawn_platform_is_flat(cfg):
    for terrain_type in ["stairs_up", "discrete_obstacles", "rough_slope"]:
        field = make_sub_terrain(terrain_type, 0.9, cfg, np.random.default_rng(0))
        n = field.shape[0]
        half = int(cfg.platform_size / cfg.horizontal_scale / 2)
        patch = field[n // 2 - half : n // 2 + half, n // 2 - half : n // 2 + half]
        assert patch.min() == patch.max(), f"{terrain_type} platform is not flat"


def test_stairs_down_mirrors_stairs_up(cfg):
    rng = np.random.default_rng(0)
    up = make_sub_terrain("stairs_up", 0.5, cfg, rng)
    down = make_sub_terrain("stairs_down", 0.5, cfg, np.random.default_rng(0))
    assert np.array_equal(down, -up)


def test_grid_assembles_and_locates_origins(cfg):
    grid = TerrainGrid(cfg, seed=0)
    n, border = cfg.cells_per_env, int(cfg.border_size / cfg.horizontal_scale)
    assert grid.heightfield.shape == (
        cfg.num_rows * n + 2 * border,
        cfg.num_cols * n + 2 * border,
    )
    x0, y0, _ = grid.origin(0, 0)
    x1, _, _ = grid.origin(1, 0)
    assert x1 - x0 == pytest.approx(cfg.terrain_length)


def test_curriculum_promotes_and_demotes(cfg):
    grid = TerrainGrid(cfg, seed=0)
    half_tile = cfg.terrain_length / 2
    levels = torch.tensor([1, 1, 1])
    # Commanded distance is deliberately below the tile length: when the two are
    # equal the promote and demote thresholds coincide and no episode can hold
    # its level.
    commanded = torch.tensor([half_tile] * 3)
    walked = torch.tensor([cfg.terrain_length, 0.1, half_tile * 0.75])

    updated = grid.update_levels(levels, walked, commanded)
    assert updated[0] == 2  # crossed half the tile -> promoted
    assert updated[1] == 0  # barely moved -> demoted
    assert updated[2] == 1  # neither -> unchanged


def test_curriculum_stays_in_range(cfg):
    grid = TerrainGrid(cfg, seed=0)
    top = cfg.num_rows - 1
    levels = torch.full((64,), top)
    walked = torch.full((64,), cfg.terrain_length)
    updated = grid.update_levels(levels, walked, walked)
    assert updated.min() >= 0 and updated.max() <= top
