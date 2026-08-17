import json

import pytest
import torch
from stub_env import StubVecEnv

from tert.data import ObservationNormalizer
from tert.envs.obs_spec import A1
from tert.eval.metrics import MetricAccumulator
from tert.eval.rollout import (
    DirectRunner,
    LatentRunner,
    StackedRunner,
    TransformerRunner,
    evaluate,
)
from tert.eval.suite import Condition, Policy, format_table, run_suite, save_results
from tert.models import TeacherActorCritic, TerrainTransformer, TransformerConfig
from tert.models.baselines import LatentPolicy, StackedActorCritic, TCNEncoder

N, CONTEXT = 4, 6


@pytest.fixture
def normalizer():
    norm = ObservationNormalizer(A1.proprio_dim)
    norm.update(torch.randn(128, A1.proprio_dim))
    return norm.freeze()


# --- metrics ----------------------------------------------------------------


def test_accumulation_stops_at_first_termination():
    """Environments auto-reset; scoring past a fall would average two episodes."""
    acc = MetricAccumulator(2)
    actions = torch.zeros(2, A1.num_actions)
    rewards = torch.ones(2)

    acc.step(actions, rewards, dones=torch.tensor([True, False]))
    acc.step(actions, rewards, dones=torch.tensor([False, False]))
    acc.step(actions, rewards, dones=torch.tensor([False, True]))

    result = acc.result()
    assert result.episode_length.tolist() == [1.0, 3.0]
    assert result.episode_return.tolist() == [1.0, 3.0]


def test_timeout_does_not_count_as_a_fall():
    acc = MetricAccumulator(2)
    acc.step(
        torch.zeros(2, A1.num_actions),
        torch.ones(2),
        dones=torch.tensor([True, True]),
        timeouts=torch.tensor([True, False]),
    )
    assert acc.result().fell.tolist() == [False, True]


def test_smoothness_measures_action_change():
    acc = MetricAccumulator(1)
    rewards, dones = torch.ones(1), torch.zeros(1, dtype=torch.bool)
    acc.step(torch.zeros(1, 2), rewards, dones)
    acc.step(torch.full((1, 2), 3.0), rewards, dones)  # |delta| = 3 per joint, 2 joints
    assert acc.result().smoothness.item() == pytest.approx(6.0 / 2)


def test_energy_is_mechanical_power():
    acc = MetricAccumulator(1)
    acc.step(
        torch.zeros(1, 2),
        torch.ones(1),
        dones=torch.zeros(1, dtype=torch.bool),
        torques=torch.tensor([[2.0, -3.0]]),
        dof_vel=torch.tensor([[1.0, 1.0]]),
    )
    assert acc.result().energy.item() == pytest.approx(5.0)  # |2| + |-3|


def test_finished_flag_ends_a_rollout_early():
    acc = MetricAccumulator(2)
    assert not acc.finished
    acc.step(torch.zeros(2, 1), torch.ones(2), dones=torch.ones(2, dtype=torch.bool))
    assert acc.finished


def test_summary_is_json_serialisable():
    acc = MetricAccumulator(3)
    acc.step(torch.zeros(3, 2), torch.ones(3), dones=torch.zeros(3, dtype=torch.bool))
    summary = acc.result(inference_ms=1.5).summary()
    json.dumps(summary)  # must not raise on tensors
    assert summary["inference_ms"] == 1.5
    assert {"episode_return", "fall_rate", "smoothness"} <= set(summary)


# --- runners ----------------------------------------------------------------


def make_transformer():
    return TerrainTransformer(
        TransformerConfig(obs_dim=A1.proprio_dim, context_len=CONTEXT, embed_dim=32, n_blocks=1)
    )


@pytest.mark.parametrize("build", ["direct", "transformer", "stacked", "latent"])
def test_every_runner_drives_the_same_loop(build, normalizer):
    """The comparison is only fair if all policies meet the env through one loop."""
    env = StubVecEnv(num_envs=N, episode_length=5)
    teacher = TeacherActorCritic()

    if build == "direct":
        runner = DirectRunner(teacher)
    elif build == "transformer":
        runner = TransformerRunner(
            make_transformer(), normalizer, N, A1.proprio_dim, A1.num_actions, CONTEXT
        )
    elif build == "stacked":
        runner = StackedRunner(
            StackedActorCritic(A1.proprio_dim, A1.num_actions, 3),
            N,
            A1.proprio_dim,
            A1.num_actions,
            3,
        )
    else:
        runner = LatentRunner(
            LatentPolicy(TCNEncoder(A1.proprio_dim, 50, 12), teacher), N, A1.proprio_dim, 50
        )

    metrics = evaluate(env, runner, num_steps=20)
    assert metrics.episode_length.shape == (N,)
    assert (metrics.episode_length > 0).all()
    assert metrics.inference_ms > 0


def test_transformer_runner_clears_context_on_termination(normalizer):
    env = StubVecEnv(num_envs=N, episode_length=3)
    runner = TransformerRunner(
        make_transformer(), normalizer, N, A1.proprio_dim, A1.num_actions, CONTEXT
    )
    evaluate(env, runner, num_steps=10, stop_when_all_done=False)
    # Env 0 terminates every 3 steps, so its timestep counter must have been reset.
    assert runner.context.timestep[0] < 10


def test_stacked_runner_resets_last_action():
    env = StubVecEnv(num_envs=N, episode_length=3)
    runner = StackedRunner(
        StackedActorCritic(A1.proprio_dim, A1.num_actions, 2), N, A1.proprio_dim, A1.num_actions, 2
    )
    evaluate(env, runner, num_steps=8, stop_when_all_done=False)
    assert runner.stacker.obs.shape == (N, 2, A1.proprio_dim)


# --- suite ------------------------------------------------------------------


def test_suite_tabulates_every_cell(tmp_path):
    policies = [
        Policy("teacher", lambda env: DirectRunner(TeacherActorCritic())),
        Policy(
            "ppo",
            lambda env: StackedRunner(
                StackedActorCritic(A1.proprio_dim, A1.num_actions, 1),
                env.num_envs,
                A1.proprio_dim,
                A1.num_actions,
                1,
            ),
        ),
    ]
    conditions = [
        Condition("stairs", lambda: StubVecEnv(num_envs=N, episode_length=4)),
        Condition("slope", lambda: StubVecEnv(num_envs=N, episode_length=7)),
    ]

    results = run_suite(policies, conditions, num_steps=15)
    assert set(results) == {"teacher", "ppo"}
    assert set(results["teacher"]) == {"stairs", "slope"}
    assert "episode_return" in results["teacher"]["stairs"]

    table = format_table(results)
    assert "teacher" in table and "stairs" in table

    path = save_results(results, tmp_path / "run.json", metadata={"steps": 15})
    saved = json.loads(path.read_text())
    assert saved["metadata"]["steps"] == 15
    assert saved["results"]["ppo"]["slope"]["fall_rate"] >= 0.0


def test_suite_reseeds_each_cell():
    """Same policy and condition twice must give identical numbers."""
    policies = [Policy("teacher", lambda env: DirectRunner(TeacherActorCritic()))]
    condition = [Condition("flat", lambda: StubVecEnv(num_envs=N, episode_length=5))]

    first = run_suite(policies, condition, num_steps=10, seed=7)
    second = run_suite(policies, condition, num_steps=10, seed=7)
    assert first["teacher"]["flat"]["episode_length"] == second["teacher"]["flat"]["episode_length"]
