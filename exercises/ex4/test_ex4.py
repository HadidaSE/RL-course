"""Sanity tests for the ex4 POMCP stack.

Run with:  python3 -m pytest exercises/ex4/test_ex4.py -q

The most important test is the *parity* test: with success probabilities
forced to 1.0 both the real course environment and the generative model are
deterministic, so their trajectories must match action-for-action.  Any
drift between the two would silently break the particle filter.
"""

from __future__ import annotations

import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from particle_filter import ParticleFilter
from pomcp import POMCPPlanner
from pomdp_env import BoxPushPOMDPEnv
from pomdp_model import BoxPushModel, joint_actions
from solution_ex4 import MAPS, ExperimentConfig, run_episode

TEST_MAP = MAPS["single"]
TEST_MAP_MULTI = MAPS["multi"]


def _make_pair(ascii_map, move_p, push_p, seed=0):
    """Builds a (real env, model) pair with matching success probabilities."""
    env = BoxPushPOMDPEnv(ascii_map, max_steps=10_000, randomize_start=False,
                          seed=seed)
    env.env.move_success_prob = move_p
    env.env.push_success_prob = push_p
    model = BoxPushModel(ascii_map, move_success_prob=move_p,
                         push_success_prob=push_p)
    return env, model


def test_model_matches_env_deterministically():
    """With p=1.0 the model must reproduce the env trajectory exactly."""
    for ascii_map in (TEST_MAP, TEST_MAP_MULTI, MAPS["ex2"]):
        env, model = _make_pair(ascii_map, move_p=1.0, push_p=1.0)
        obs, _ = env.reset()
        state = model.initial_state(model.map_agent_starts)
        assert env.true_state() == state
        assert obs == model.observe(state)

        rng = random.Random(7)
        for _ in range(300):
            joint = rng.choice(joint_actions(model.n_agents))
            actions = {a: joint[i] for i, a in enumerate(env.agents)}
            obs, _, terminated, _, _ = env.step(actions)
            state, model_obs, _, model_done = model.step(state, joint, rng)
            assert env.true_state() == state, f"state drift on {ascii_map}"
            assert obs == model_obs, f"observation drift on {ascii_map}"
            assert terminated == model_done
            if terminated:
                break


def test_observation_function_matches_under_stochastic_dynamics():
    """The wrapper's window must equal the model's for the true state."""
    env, model = _make_pair(TEST_MAP_MULTI, move_p=0.8, push_p=0.8, seed=3)
    env.randomize_start = True
    obs, _ = env.reset()
    rng = random.Random(11)
    for _ in range(200):
        assert obs == model.observe(env.true_state())
        joint = rng.choice(joint_actions(model.n_agents))
        obs, _, terminated, truncated, _ = env.step(
            {a: joint[i] for i, a in enumerate(env.agents)}
        )
        if terminated or truncated:
            break


def test_egocentric_window_semantics():
    """Alternative B: the window is the 3x3 block centred on the agent."""
    model = BoxPushModel(TEST_MAP)
    # Agent at (3, 3) on the 'single' map: the box sits at (3, 2), directly
    # above it, so it must appear in the top-middle cell of the window.
    win = model.window((3, 3), frozenset({(3, 2)}))
    #        (2,2)(3,2)(4,2)   free BOX  free
    #        (2,3)(3,3)(4,3) = free free free   <- agent in the centre
    #        (2,4)(3,4)(4,4)   free free free
    assert win == (0, 2, 0, 0, 0, 0, 0, 0, 0)
    # Agent in the top-left playable corner: the wall border must show up as
    # WALL along the top row and the left column.
    win_corner = model.window((1, 1), frozenset())
    assert win_corner == (1, 1, 1, 1, 0, 0, 1, 0, 0)


def test_particle_filter_tracks_true_state():
    """The true state should (almost) always survive the belief update."""
    np.random.seed(5)
    env, model = _make_pair(TEST_MAP, move_p=0.8, push_p=0.8, seed=5)
    env.randomize_start = True
    obs, _ = env.reset()
    pf = ParticleFilter(model, n_particles=300, rng=random.Random(5))
    pf.initialize()
    pf.condition_on_observation(obs)

    rng = random.Random(13)
    hits, total = 0, 0
    for _ in range(60):
        joint = rng.choice(joint_actions(model.n_agents))
        obs, _, terminated, truncated, _ = env.step(
            {a: joint[i] for i, a in enumerate(env.agents)}
        )
        pf.update(joint, obs)
        total += 1
        if env.true_state() in pf.particles:
            hits += 1
        if terminated or truncated:
            break
    assert hits / total > 0.9, f"true state kept in belief only {hits}/{total}"


def test_particle_filter_survives_depletion():
    """An impossible observation must trigger reinvigoration, not a crash."""
    model = BoxPushModel(TEST_MAP)
    pf = ParticleFilter(model, n_particles=50, rng=random.Random(1))
    pf.initialize()
    impossible = ((1,) * 9,)  # walls everywhere — matches no free cell
    stats = pf.update((0,), impossible)
    assert len(pf.particles) == pf.n_particles
    assert stats["reinvigorated"] > 0


def test_pomcp_budget_is_enforced():
    """plan() must return within the wall-clock budget (small tolerance)."""
    model = BoxPushModel(TEST_MAP)
    pf = ParticleFilter(model, n_particles=100, rng=random.Random(2))
    pf.initialize()
    planner = POMCPPlanner(model, rng=random.Random(2))
    budget = 0.5
    t0 = time.monotonic()
    planner.plan(pf.particles, time_budget=budget)
    elapsed = time.monotonic() - t0
    assert elapsed < budget + 0.25, f"budget overrun: {elapsed:.2f}s"
    assert planner.last_stats["simulations"] > 100


def test_online_loop_solves_single_scenario():
    """A short-budget POMCP run should solve the small single-agent map."""
    config = ExperimentConfig(n_particles=300, max_steps=120)
    result = run_episode("single", time_budget=0.3, seed=42, config=config)
    assert result["solved"], f"episode not solved in {result['steps']} steps"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name} ...", flush=True)
            fn()
            print("  OK")
    print("All tests passed.")
