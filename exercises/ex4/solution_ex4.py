"""Assignment 4 — POMDP planning with POMCP in the Box Pushing environment.

Online planning loop (per the assignment):

    obs, info = env.reset()
    particles = init_uniform_particles(free_cells)
    done = False
    while not done:
        action = run_pomcp(particles, time_budget=time_limit)   # 1. plan
        obs, reward, term, trunc, info = env.step(action)        # 2. act
        particles = update_particles(particles, action, obs)     # 3. update
        done = term or trunc

Experiments: for each scenario (single agent / two robots) and each decision
budget (1 s / 20 s), the loop is run 30 times and the mean and standard
deviation of the number of environment steps to solve the task are reported.

Usage examples:
    python3 exercises/ex4/solution_ex4.py                       # full run
    python3 exercises/ex4/solution_ex4.py --quick               # smoke test
    python3 exercises/ex4/solution_ex4.py --budgets 1 --jobs 4  # s1 only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from particle_filter import ParticleFilter
from pomcp import POMCPPlanner
from pomdp_env import BoxPushPOMDPEnv
from pomdp_model import BoxPushModel

logger = logging.getLogger("rl_ex4")

# ---------------------------------------------------------------------------
# Scenario maps.  'A' marks how many agents exist; with randomized starts the
# actual initial locations are re-drawn uniformly per run (a sample from b0).
# ---------------------------------------------------------------------------
MAPS: Dict[str, List[str]] = {
    # Single robot, one small box, one goal.
    "single": [
        "WWWWWWW",
        "W A   W",
        "W  B  W",
        "W     W",
        "W  G  W",
        "WWWWWWW",
    ],
    # Two robots, two small boxes, two goals.
    "multi": [
        "WWWWWWW",
        "W A A W",
        "W B B W",
        "W     W",
        "W G G W",
        "WWWWWWW",
    ],
    # The full Assignment 1/2 map (heavy box requires a joint push); much
    # harder under location uncertainty — provided for completeness.
    "ex2": [
        "WWWWWWWW",
        "W  AA  W",
        "W B C  W",
        "W      W",
        "W   B  W",
        "W G G GW",
        "WWWWWWWW",
    ],
}


@dataclass
class ExperimentConfig:
    """Hyperparameters and runtime settings for the ex4 experiments."""

    scenarios: List[str] = field(default_factory=lambda: ["single", "multi"])
    budgets: List[float] = field(default_factory=lambda: [1.0, 20.0])
    n_runs: int = 30
    n_particles: int = 500
    exploration_c: float = 1.0
    max_depth: int = 60
    gamma: float = 0.95
    max_steps: int = 200
    rollout_epsilon: float = 0.2
    obs_mode: str = "egocentric"  # "egocentric" (Alt. B) or "north" (Alt. A)
    base_seed: int = 0
    jobs: int = 1


# ---------------------------------------------------------------------------
# One episode of the online planning loop
# ---------------------------------------------------------------------------

def run_episode(
    scenario: str,
    time_budget: float,
    seed: int,
    config: ExperimentConfig,
    verbose: bool = False,
) -> dict:
    """Runs one full episode of POMCP online planning on the real env.

    Args:
        scenario: Key into ``MAPS``.
        time_budget: Per-decision wall-clock budget in seconds.
        seed: Seed for this run (env dynamics, start location, planner and
            particle-filter randomness).
        config: Experiment hyperparameters.
        verbose: If True, log per-step diagnostics at DEBUG level.

    Returns:
        Dict with ``steps`` (env steps taken), ``solved`` (True iff the
        goal configuration was reached before truncation) and ``seconds``
        (wall-clock episode duration).
    """
    ascii_map = MAPS[scenario]
    model = BoxPushModel(ascii_map, obs_mode=config.obs_mode)
    env = BoxPushPOMDPEnv(
        ascii_map,
        max_steps=config.max_steps,
        randomize_start=True,
        seed=seed,
        obs_mode=config.obs_mode,
    )
    planner = POMCPPlanner(
        model,
        gamma=config.gamma,
        exploration_c=config.exploration_c,
        max_depth=config.max_depth,
        rng=random.Random(seed + 1_000_003),
    )
    planner.rollout_policy.epsilon = config.rollout_epsilon
    pf = ParticleFilter(
        model, n_particles=config.n_particles, rng=random.Random(seed + 2_000_003)
    )

    t_start = time.monotonic()
    obs, _ = env.reset()
    pf.initialize()
    # The initial observation arrives before any action; condition b0 on it.
    pf.condition_on_observation(obs)

    steps = 0
    solved = False
    done = False
    while not done:
        joint_action = planner.plan(pf.particles, time_budget=time_budget)
        actions = {agent: joint_action[i] for i, agent in enumerate(env.agents)}
        obs, reward, terminated, truncated, _ = env.step(actions)
        stats = pf.update(joint_action, obs)
        steps += 1
        if verbose:
            logger.debug(
                "step=%d action=%s reward=%.1f pf_accepted=%d/%d "
                "reinvig=%d sims=%d",
                steps,
                joint_action,
                reward,
                stats["accepted"],
                stats["attempts"],
                stats["reinvigorated"],
                planner.last_stats.get("simulations", -1),
            )
        if terminated:
            solved = True
        done = terminated or truncated

    return {
        "steps": steps,
        "solved": solved,
        "seconds": time.monotonic() - t_start,
    }


def _episode_worker(args: Tuple[str, float, int, ExperimentConfig]) -> dict:
    """Top-level worker so episodes can run in a multiprocessing pool."""
    scenario, budget, seed, config = args
    np.random.seed(seed)  # the course env draws from the global numpy RNG
    return run_episode(scenario, budget, seed, config)


# ---------------------------------------------------------------------------
# Experiment harness
# ---------------------------------------------------------------------------

def run_experiment(
    scenario: str, time_budget: float, config: ExperimentConfig
) -> dict:
    """Runs ``config.n_runs`` episodes and aggregates step statistics.

    Args:
        scenario: Key into ``MAPS``.
        time_budget: Per-decision budget in seconds.
        config: Experiment hyperparameters.

    Returns:
        Dict with mean/std of steps, solve rate and per-run raw results.
    """
    tasks = [
        (scenario, time_budget, config.base_seed + 10_000 * i, config)
        for i in range(config.n_runs)
    ]
    results: List[dict] = []
    if config.jobs > 1:
        with Pool(config.jobs) as pool:
            for i, res in enumerate(pool.imap(_episode_worker, tasks)):
                results.append(res)
                logger.info(
                    "  [%s | %gs] run %d/%d — steps=%d solved=%s (%.0fs)",
                    scenario, time_budget, i + 1, config.n_runs,
                    res["steps"], res["solved"], res["seconds"],
                )
    else:
        for i, task in enumerate(tasks):
            res = _episode_worker(task)
            results.append(res)
            logger.info(
                "  [%s | %gs] run %d/%d — steps=%d solved=%s (%.0fs)",
                scenario, time_budget, i + 1, config.n_runs,
                res["steps"], res["solved"], res["seconds"],
            )

    steps = np.array([r["steps"] for r in results], dtype=float)
    solved = np.array([r["solved"] for r in results])
    return {
        "scenario": scenario,
        "budget": time_budget,
        "obs_mode": config.obs_mode,
        "mean_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "solve_rate": float(np.mean(solved)),
        "n_runs": len(results),
        "runs": results,
    }


def print_summary(all_results: List[dict]) -> None:
    """Logs the final results table required by the assignment."""
    logger.info("")
    logger.info("=" * 68)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 68)
    logger.info(
        "%-12s %-10s %12s %12s %12s",
        "Scenario", "Budget", "Mean steps", "Std steps", "Solve rate",
    )
    logger.info("-" * 68)
    for res in all_results:
        logger.info(
            "%-12s %-10s %12.2f %12.2f %11.0f%%",
            res["scenario"],
            f"{res['budget']:g}s",
            res["mean_steps"],
            res["std_steps"],
            100 * res["solve_rate"],
        )
    logger.info("=" * 68)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", nargs="+", default=["single", "multi"],
                        choices=list(MAPS), help="Scenarios to run.")
    parser.add_argument("--budgets", nargs="+", type=float, default=[1.0, 20.0],
                        help="Per-decision time budgets in seconds.")
    parser.add_argument("--runs", type=int, default=30,
                        help="Episodes per (scenario, budget) pair.")
    parser.add_argument("--particles", type=int, default=500,
                        help="Particle filter size N.")
    parser.add_argument("--c", type=float, default=1.0,
                        help="UCB1 exploration constant.")
    parser.add_argument("--depth", type=int, default=60,
                        help="POMCP search / rollout horizon.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor.")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Episode truncation limit.")
    parser.add_argument("--obs", default="egocentric",
                        choices=["egocentric", "north"],
                        help="Observation function: 'egocentric' = "
                             "Alternative B (3x3 window centred on the "
                             "agent); 'north' = Alternative A (3x3 window "
                             "immediately north of the agent).")
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Parallel episodes (keep <= physical cores so "
                             "each decision still gets a full core).")
    parser.add_argument("--quick", action="store_true",
                        help="Tiny smoke test (short budgets, few runs).")
    parser.add_argument("--out", default=None,
                        help="Output prefix for results files (default: "
                             "results in this directory).")
    return parser.parse_args(argv)


def setup_logging(log_path: str) -> None:
    """Console shows INFO progress; the file keeps DEBUG traces."""
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(console)
    logger.addHandler(file_handler)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    out_prefix = args.out or os.path.join(here, "results")
    setup_logging(out_prefix + ".log")

    config = ExperimentConfig(
        scenarios=args.scenarios,
        budgets=args.budgets,
        n_runs=args.runs,
        n_particles=args.particles,
        exploration_c=args.c,
        max_depth=args.depth,
        gamma=args.gamma,
        max_steps=args.max_steps,
        obs_mode=args.obs,
        base_seed=args.seed,
        jobs=args.jobs,
    )
    if args.quick:
        config.budgets = [0.1]
        config.n_runs = 3

    logger.info("Config: %s", config)
    all_results = []
    for scenario in config.scenarios:
        for budget in config.budgets:
            logger.info(
                "Running scenario=%s budget=%gs (%d runs)...",
                scenario, budget, config.n_runs,
            )
            all_results.append(run_experiment(scenario, budget, config))

    print_summary(all_results)
    with open(out_prefix + ".json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Raw results written to %s.json", out_prefix)


if __name__ == "__main__":
    main()
